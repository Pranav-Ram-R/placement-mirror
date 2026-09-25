"""Local web server: serves the UI and runs the vision and audio pipelines per WebSocket session.

- GET /             the UI in app/ui/static/
- GET /api/status   ModelRunner.status(): the real compute unit and load path per model
- WS  /ws           one session. Binary messages carry frames in (app.vision.pipeline
                    parse_frame). Text messages carry JSON commands in ({"type":
                    "calibrate"} or {"type": "stop"}) and JSON events out.

Models load once at startup through ModelRunner. Binds to 127.0.0.1 only and makes no
network calls. One session at a time, because all sessions would share the same models
and CPU. Audio comes from the default microphone through sounddevice (app.audio.capture),
or from --audio-file played in real time.

When a session ends (Stop in the UI or the socket closes) the per stage p50/p95 table and
the per segment audio timings are printed. With --stats-dir they are saved there as
Measurement records. Transcript text is never saved.

Usage: python -m app.server [--port 8000] [--stats-dir DIR] [--label "what the input was"]
                            [--audio-file clip.wav|clip.npy] [--no-audio]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.audio.pipeline import AUDIO_MODELS, AudioPipeline
from app.config import AUDIO
from app.runtime.priority import AsrPriority, disable_power_throttling
from app.runtime.runner import CPU_ONLY, ModelRunner
from app.vision.pipeline import VISION_MODELS, VisionPipeline

STATIC = Path(__file__).resolve().parent / "ui" / "static"
SETTINGS = {"stats_dir": None, "label": "", "audio_file": None, "audio": True}
STATE: dict = {"runner": None, "session_lock": threading.Lock()}


def cpu_fallback_notice(status: dict) -> str | None:
    """Rule 6: a visible notice when a model that should run on the NPU runs on CPU."""
    fallen = {n: s for n, s in status.items() if s["compute_unit"] != "NPU" and n not in CPU_ONLY}
    if not fallen:
        return None
    reasons = sorted({s["reason"] for s in fallen.values()})
    return f"{', '.join(fallen)} running on CPU. Reason: {' '.join(reasons)}"


def status_payload() -> dict:
    status = STATE["runner"].status()
    return {"models": status, "notice": cpu_fallback_notice(status)}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"Windows power throttling disabled for this process: {disable_power_throttling()}", flush=True)
    runner = ModelRunner()
    for name in VISION_MODELS + (AUDIO_MODELS if SETTINGS["audio"] else ()):
        st = runner.load(name)
        print(f"loaded {name}: {st.compute_unit} ({st.path}). {st.reason()}", flush=True)
    STATE["runner"] = runner
    notice = cpu_fallback_notice(runner.status())
    if notice:
        print(f"NOTICE: {notice}", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(status_payload())


def measurement_source():
    from benchmarks.schema import Source

    return Source.LOCAL_X86_CPU if platform.machine().upper() in ("AMD64", "X86_64") else Source.PHYSICAL_SNAPDRAGON


def print_report(report: dict) -> None:
    print(f"\nSession ({report['mode']}, face up to {report['face_max_fps']:g} fps, pose {report['pose_fps']:g} fps): "
          f"{report['duration_s']:.1f} s, frames received {report['frames_received']}, "
          f"rate limited {report['frames_rate_limited']}, pose runs {report['pose_runs']}, "
          f"pose skipped for ASR {report['pose_skipped_for_asr']}, "
          f"processed {report['frames_processed']}, replaced in mailbox {report['frames_replaced_in_mailbox']}, "
          f"dropped in browser {report.get('browser', {}).get('dropped_busy', 'n/a')}, "
          f"face detector runs {report['face_detector_runs']}")
    print("| stage | count | p50 ms | p95 ms |")
    print("|---|---|---|---|")
    for stage, s in report["stages"].items():
        print(f"| {stage} | {s['count']} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} |")
    sys.stdout.flush()


def save_report(report: dict, status: dict, stats_dir: Path, stamp: str) -> Path:
    from benchmarks.schema import Derived, Measurement, save_json

    src = measurement_source()
    stats_dir.mkdir(parents=True, exist_ok=True)
    units = {n: s["compute_unit"] for n, s in status.items() if n in VISION_MODELS}
    stage_model = {"face_detector": "face_detector", "face_landmark": "face_landmark",
                   "pose_detector": "pose_detector", "pose_landmark": "pose_landmark"}
    session = (f"{SETTINGS['label'] + '. ' if SETTINGS['label'] else ''}"
               f"live session {report['duration_s']:.1f} s, {report['frames_processed']} frames processed of "
               f"{report['frames_received']} received, browser {report.get('browser', {})}, "
               f"ModelRunner compute units {units}, CPU {platform.processor()}")
    records = []
    for stage, s in report["stages"].items():
        model = stage_model.get(stage)
        runtime = ("onnxruntime" if model else "onnxruntime and numpy" if stage in ("pose", "total")
                   else "python" if stage == "queue_wait" else "numpy")
        unit = units.get(model, "CPU") if model else ("CPU" if stage in ("roi_warp", "pose_warp", "head_pose", "queue_wait")
                                                     else "+".join(sorted(set(units.values()) | {"CPU"})))
        for q in (50, 95):
            records.append(Measurement(
                model=f"vision_pipeline.{stage}", metric=f"stage_time_p{q}", value=s[f"p{q}_ms"], unit="ms",
                source=src, runtime=runtime, compute_unit=unit,
                precision=status[model]["precision"] if model else ("float32" if "warp" in stage else "unknown"),
                ort_version=STATE["runner"].ort.__version__,
                notes=f"{s['count']} samples, numpy.percentile linear. {session}"))
    common = dict(source=src, runtime="onnxruntime and numpy", compute_unit="+".join(sorted(set(units.values()))),
                  precision="float32", ort_version=STATE["runner"].ort.__version__, notes=session)
    processed = Measurement(model="vision_pipeline", metric="frames_processed", value=report["frames_processed"],
                            unit="frames", **common)
    duration = Measurement(model="vision_pipeline", metric="session_duration", value=report["duration_s"], unit="s",
                           **common)
    records += [processed, duration, Derived(
        name="vision_pipeline processed fps", value=report["frames_processed"] / report["duration_s"], unit="fps",
        formula="frames_processed / session_duration", inputs=[processed, duration])]
    path = stats_dir / f"{stamp}_vision_session.json"
    save_json(records, path)
    (stats_dir / f"{stamp}_vision_session_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def print_audio_report(report: dict) -> None:
    vad = report["vad_call_ms"]
    print(f"\nAudio: {report['source']}, {report['duration_s']:.1f} s, {len(report['segments'])} segments, "
          f"{report['words_total']} words, {report['filler_count']} fillers, {len(report['pauses'])} pause events, "
          f"VAD call p50 {vad.get('p50', float('nan')):.3f} ms p95 {vad.get('p95', float('nan')):.3f} ms")
    print("| segment | audio s | closed by | VAD latency ms | encoder ms | decoder ms per token | tokens | end to end ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in report["segments"]:
        print(f"| {r['index']} | {r['audio_s']:.2f} | {r['closed_by']} | {r['vad_latency_ms']:.1f} | {r['encoder_ms']:.1f} "
              f"| {r['decoder_ms_per_token']:.2f} | {r['token_count']} | {r['end_to_end_delay_ms']:.1f} |")
    sys.stdout.flush()


AUDIO_METRICS = {  # report key -> (model, metric, unit, model whose placement and precision apply, how)
    "vad_latency_ms": ("audio_pipeline.vad", "vad_latency", "ms", "silero_vad",
                       "segment closed by the VAD minus the end of its last speech block, includes the "
                       f"{AUDIO.segment_end_silence_ms} ms end of segment silence"),
    "encoder_ms": ("whisper_tiny_encoder", "encoder_time", "ms", "whisper_tiny_encoder",
                   "wall time of the encoder session.run"),
    "decoder_ms_per_token": ("whisper_tiny_decoder", "decoder_time_per_token", "ms", "whisper_tiny_decoder",
                             "decoder wall time / decoder calls, one token position per call including the "
                             "forced prefix"),
    "token_count": ("whisper_tiny_decoder", "token_count", "tokens", "whisper_tiny_decoder",
                    "generated tokens without the prefix and end of text"),
    "end_to_end_delay_ms": ("audio_pipeline.end_to_end", "end_to_end_delay", "ms", "whisper_tiny_decoder",
                            "transcript ready minus the end of the last speech block"),
}


def save_audio_report(report: dict, status: dict, stats_dir: Path, stamp: str) -> Path:
    from benchmarks.schema import Measurement, save_json

    src = measurement_source()
    stats_dir.mkdir(parents=True, exist_ok=True)
    units = {n: status[n]["compute_unit"] for n in AUDIO_MODELS}
    base = (f"{SETTINGS['label'] + '. ' if SETTINGS['label'] else ''}audio input {report['source']}, "
            f"session {report['duration_s']:.1f} s, {len(report['segments'])} segments, use_prompt "
            f"{report['use_prompt']}, ModelRunner compute units {units}, CPU {platform.processor()}")
    ort = STATE["runner"].ort.__version__
    records = []
    for key, (model, metric, unit, placed, how) in AUDIO_METRICS.items():
        st = status[placed]
        common = dict(model=model, unit=unit, source=src, runtime=st["runtime"], compute_unit=st["compute_unit"],
                      precision=st["precision"] or "unknown", ort_version=ort)
        for r in report["segments"]:
            if r[key] == r[key]:  # skip NaN
                records.append(Measurement(metric=metric, value=r[key], notes=(
                    f"segment {r['index']}, {r['audio_s']:.2f} s of audio, closed by {r['closed_by']}, "
                    f"{r['decoder_calls']} decoder calls, stopped at {r['stopped']}. {how}. {base}"), **common))
        summary = report["summary"][key]
        if summary.get("count"):
            for q in (50, 95):
                records.append(Measurement(metric=f"{metric}_p{q}", value=summary[f"p{q}"], notes=(
                    f"p{q} over {summary['count']} segments, numpy.percentile linear. {how}. {base}"), **common))
    vad = report["vad_call_ms"]
    if vad.get("count"):
        st = status["silero_vad"]
        for q in (50, 95):
            records.append(Measurement(
                model="silero_vad", metric=f"inference_time_p{q}", value=vad[f"p{q}"], unit="ms", source=src,
                runtime=st["runtime"], compute_unit=st["compute_unit"], precision=st["precision"] or "unknown",
                ort_version=ort, notes=f"p{q} over {vad['count']} 32 ms blocks, wall time of session.run. {base}"))
    path = stats_dir / f"{stamp}_audio_session.json"
    save_json(records, path)
    no_text = {**report, "segments": [{k: v for k, v in r.items() if k not in ("text", "fillers")}
                                      for r in report["segments"]]}
    (stats_dir / f"{stamp}_audio_session_report.json").write_text(json.dumps(no_text, indent=2) + "\n", encoding="utf-8")
    return path


def finish(vision: VisionPipeline, audio: AudioPipeline | None, browser: dict) -> dict:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stats_dir = Path(SETTINGS["stats_dir"]) if SETTINGS["stats_dir"] else None
    status = STATE["runner"].status()
    report = vision.report()
    report["browser"] = browser
    if report["frames_processed"]:
        print_report(report)
        if stats_dir:
            print(f"saved stage measurements to {save_report(report, status, stats_dir, stamp)}", flush=True)
    if audio is not None:
        audio_report = audio.report()
        print_audio_report(audio_report)
        if stats_dir and audio_report["segments"]:
            print(f"saved audio measurements to {save_audio_report(audio_report, status, stats_dir, stamp)}", flush=True)
        report["audio"] = {**audio_report,
                           "segments": [{k: v for k, v in r.items() if k != "text"} for r in audio_report["segments"]]}
    return report


def load_audio_file(path: Path):
    import numpy as np

    if path.suffix == ".npy":
        return np.load(path).astype(np.float32).reshape(-1)
    if path.suffix == ".npz":
        z = np.load(path)
        return z[z.files[0]].astype(np.float32).reshape(-1)
    import wave

    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise SystemExit(f"{path}: need 16 kHz mono 16 bit PCM WAV")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def audio_source_factory():
    path = SETTINGS["audio_file"]
    if path is None:
        return None  # microphone
    from app.audio.capture import FileSource

    audio = load_audio_file(Path(path))
    return lambda put: FileSource(put, audio, f"file {Path(path).name} played in real time")


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    lock = STATE["session_lock"]
    if not lock.acquire(blocking=False):
        await ws.send_json({"type": "error", "message": "Another session is already running. Close it first."})
        await ws.close()
        return
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(events.put_nowait, event)

    async def send_events() -> None:
        while True:
            await ws.send_json(await events.get())

    priority = AsrPriority()
    vision = VisionPipeline(STATE["runner"], emit, priority=priority)
    await ws.send_json({"type": "status", **status_payload()})
    sender = asyncio.create_task(send_events())
    vision.start()
    audio = None
    if SETTINGS["audio"]:
        try:
            audio = AudioPipeline(STATE["runner"], emit, source_factory=audio_source_factory(), priority=priority)
            audio.start()
            emit({"type": "audio", "state": "running", "source": audio.source.describe()})
        except Exception as e:  # noqa: BLE001  video keeps running without audio
            audio = None
            emit({"type": "error", "message": f"Audio could not start: {type(e).__name__}: {e}"})
    browser: dict = {}
    stopped = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                print(f"session closed by the client, code {msg.get('code')} {msg.get('reason') or ''}", flush=True)
                break
            if msg.get("bytes") is not None:
                vision.submit(msg["bytes"], time.monotonic(), time.perf_counter())
            elif msg.get("text") is not None:
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "calibrate":
                    vision.calibrate(time.monotonic())
                elif cmd.get("type") == "stop":
                    browser = cmd.get("browser", {})
                    await asyncio.to_thread(vision.stop)
                    if audio is not None:
                        await asyncio.to_thread(audio.stop)
                    report = await asyncio.to_thread(finish, vision, audio, browser)
                    stopped = True
                    emit({"type": "stats", **report})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if not stopped:
            await asyncio.to_thread(vision.stop)
            if audio is not None:
                await asyncio.to_thread(audio.stop)
            await asyncio.to_thread(finish, vision, audio, browser)
        await asyncio.sleep(0.1)  # let queued events go out
        sender.cancel()
        lock.release()


app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stats-dir", type=Path, help="save each session's timings here as Measurements")
    ap.add_argument("--label", default="", help="what the session input was, saved in the Measurement notes")
    ap.add_argument("--audio-file", type=Path, help="play this 16 kHz mono recording instead of the microphone")
    ap.add_argument("--no-audio", action="store_true", help="run the vision pipeline only")
    args = ap.parse_args()
    SETTINGS.update(stats_dir=args.stats_dir, label=args.label, audio_file=args.audio_file, audio=not args.no_audio)
    # No keepalive pings: the socket is on localhost and the browser closes it with the tab.
    # With uvicorn's default (20 s ping, 20 s pong timeout) sessions closed at 40 s while
    # frames were streaming (2026-09-25 test), cause not found.
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws="wsproto", ws_ping_interval=None, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
