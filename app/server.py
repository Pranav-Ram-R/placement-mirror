"""Local web server: serves the UI and runs one interview session per WebSocket.

- GET /               the UI in app/ui/static/
- GET /api/status     ModelRunner.status(): the real compute unit and load path per model
- GET /api/questions  the question bank (questions/bank.json)
- GET /api/config     replay mode settings for the browser
- GET /replay/video   the replay video file (replay mode only)
- GET /api/sessions   saved answers with the history trend metrics (app.storage.history)
- GET /api/sessions/{id}/report  one answer's report (app.analysis.report)
- POST /api/eval/recordings/{id}/video  the browser's WebM of one answer (--record-eval only)
- WS  /ws             one session (app.session.controller). Binary messages carry frames in
                      (app.vision.pipeline parse_frame). Text messages carry JSON commands
                      in (select_question, calibrate, start_answer, stop_answer, stop) and
                      JSON events out.

Replay mode (--replay VIDEO WAV, a development flag): the browser plays VIDEO in a video
element as its frame source instead of the camera, and the server plays WAV instead of
the microphone. Both start on the same start answer command.

Eval recording mode (--record-eval, a development flag, off by default): each answer's
microphone audio (the same sounddevice stream the pipeline uses) and the browser's
MediaRecorder WebM of the camera stream are saved to eval/recordings/<session id>/
(gitignored), with the question id and both streams' start times. The page shows a red
RECORDING FOR EVAL banner. Without the flag the upload route answers 404 and no raw media
is written.

Models load once at startup through ModelRunner. Binds to 127.0.0.1 only and makes no
network calls. One session at a time, because all sessions would share the same models
and CPU. Audio comes from the default microphone through sounddevice (app.audio.capture),
or from --audio-file played in real time.

When a session ends (Stop in the UI or the socket closes) the per stage p50/p95 table and
the per segment audio timings are printed. With --stats-dir they are saved there as
Measurement records. Transcript text is never saved.

Usage: python -m app.server [--port 8000] [--stats-dir DIR] [--label "what the input was"]
                            [--audio-file clip.wav|clip.npy] [--no-audio]
                            [--replay VIDEO WAV] [--data-dir DIR] [--record-eval]
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

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analysis.feedback import guideline_dict
from app.audio.pipeline import AUDIO_MODELS, AudioPipeline
from app.config import AUDIO, REPORT, SESSION
from app.runtime.priority import disable_power_throttling
from app.runtime.runner import CPU_ONLY, ModelRunner
from app.session.controller import SessionController
from app.session.questions import default_bank
from app.storage import history, recordings
from app.vision.pipeline import VISION_MODELS, VisionPipeline

STATIC = Path(__file__).resolve().parent / "ui" / "static"
SETTINGS = {"stats_dir": None, "label": "", "audio_file": None, "audio": True, "replay": None, "data_dir": None,
            "record_eval": False}
VIDEO_TYPES = {".webm": "video/webm", ".mp4": "video/mp4", ".ogv": "video/ogg"}
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


@app.get("/api/questions")
def api_questions() -> JSONResponse:
    bank = default_bank()
    return JSONResponse({k: bank[k] for k in ("status", "categories", "questions")})


def replay_info() -> dict | None:
    if not SETTINGS["replay"]:
        return None
    video, wav = SETTINGS["replay"]
    return {"video_url": "/replay/video", "video": Path(video).name, "audio": Path(wav).name}


@app.get("/api/config")
def api_config() -> JSONResponse:
    return JSONResponse({"replay": replay_info(), "auto_stop_extra_s": SESSION.auto_stop_extra_s,
                         "record_eval": SETTINGS["record_eval"]})


@app.post("/api/eval/recordings/{session_id}/video")
async def api_eval_video(session_id: str, request: Request, start_epoch_ms: float | None = None,
                         stop_epoch_ms: float | None = None, mime: str = "video/webm") -> JSONResponse:
    if not SETTINGS["record_eval"]:
        raise HTTPException(status_code=404, detail="eval recording is off")
    folder = recordings.RECORDINGS_DIR / session_id
    if not history.SESSION_ID_RE.fullmatch(session_id) or not (folder / recordings.META).exists():
        raise HTTPException(status_code=404, detail=f"no eval recording {session_id}")
    data = await request.body()
    meta = await asyncio.to_thread(recordings.save_video, folder, data, start_epoch_ms=start_epoch_ms,
                                   stop_epoch_ms=stop_epoch_ms, mime=mime)
    print(f"eval recording {session_id}: video {meta['video'].get('duration_s')} s, audio "
          f"{(meta.get('audio') or {}).get('duration_s')} s, difference {meta.get('duration_difference_ms')} ms",
          flush=True)
    return JSONResponse(meta)


@app.get("/api/sessions")
def api_sessions() -> JSONResponse:
    return JSONResponse({"sessions": history.list_sessions(sessions_dir()), "trend_metrics": history.TREND_METRICS,
                         "guidelines": [guideline_dict(g) for g in REPORT.guidelines]})


@app.get("/api/sessions/{session_id}/report")
def api_report(session_id: str) -> JSONResponse:
    try:
        return JSONResponse(history.load_report(session_id, sessions_dir()))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no saved answer {session_id}") from None


@app.get("/replay/video")
def replay_video() -> FileResponse:
    if not SETTINGS["replay"]:
        raise HTTPException(status_code=404, detail="not in replay mode")
    path = Path(SETTINGS["replay"][0])
    return FileResponse(path, media_type=VIDEO_TYPES.get(path.suffix.lower(), "application/octet-stream"))


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


def finish(vision: VisionPipeline, audio_runs: list, browser: dict) -> dict:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stats_dir = Path(SETTINGS["stats_dir"]) if SETTINGS["stats_dir"] else None
    status = STATE["runner"].status()
    report = vision.report()
    report["browser"] = browser
    if report["frames_processed"]:
        print_report(report)
        if stats_dir:
            print(f"saved stage measurements to {save_report(report, status, stats_dir, stamp)}", flush=True)
    report["audio_runs"] = []
    for i, audio in enumerate(audio_runs):
        audio_report = audio.report()
        print_audio_report(audio_report)
        if stats_dir and audio_report["segments"]:
            path = save_audio_report(audio_report, status, stats_dir, f"{stamp}_answer{i + 1}")
            print(f"saved audio measurements to {path}", flush=True)
        report["audio_runs"].append({**audio_report, "segments": [
            {k: v for k, v in r.items() if k not in ("text", "fillers")} for r in audio_report["segments"]]})
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
    path = SETTINGS["replay"][1] if SETTINGS["replay"] else SETTINGS["audio_file"]
    if path is None:
        return None  # microphone
    from app.audio.capture import FileSource

    audio = load_audio_file(Path(path))
    return lambda put: FileSource(put, audio, f"file {Path(path).name} played in real time")


def audio_factory():
    """Builds the audio pipeline for each answer, or None when audio is off."""
    if not SETTINGS["audio"]:
        return None
    source = audio_source_factory()
    return lambda emit, priority, recorder=None: AudioPipeline(STATE["runner"], emit, source_factory=source,
                                                               priority=priority, recorder=recorder)


def sessions_dir() -> Path | None:
    return Path(SETTINGS["data_dir"]) / "sessions" if SETTINGS["data_dir"] else None


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

    controller = SessionController(STATE["runner"], emit, sessions_dir=sessions_dir(), audio_factory=audio_factory(),
                                   replay=replay_info(),
                                   eval_recordings=recordings.RECORDINGS_DIR if SETTINGS["record_eval"] else None)
    await ws.send_json({"type": "status", **status_payload()})
    sender = asyncio.create_task(send_events())
    controller.start()
    browser: dict = {}
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                print(f"session closed by the client, code {msg.get('code')} {msg.get('reason') or ''}", flush=True)
                break
            if msg.get("bytes") is not None:
                controller.vision.submit(msg["bytes"], time.monotonic(), time.perf_counter())
            elif msg.get("text") is not None:
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "stop":  # end of the connection, with the browser's counters
                    browser = cmd.get("browser", {})
                    break
                controller.command(cmd, time.monotonic())
    except WebSocketDisconnect:
        pass
    finally:
        await asyncio.to_thread(controller.close)
        report = await asyncio.to_thread(finish, controller.vision, controller.audio_runs, browser)
        emit({"type": "stats", **report})
        await asyncio.sleep(0.2)  # let queued events go out
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
    ap.add_argument("--replay", nargs=2, type=Path, metavar=("VIDEO", "WAV"),
                    help="development: play VIDEO in the browser and WAV instead of the microphone")
    ap.add_argument("--data-dir", type=Path, help="session data folder (default %%LOCALAPPDATA%%\\PlacementMirror)")
    ap.add_argument("--record-eval", action="store_true",
                    help="development: save each answer's camera WebM and microphone WAV to eval/recordings")
    args = ap.parse_args()
    if args.record_eval and (args.replay or args.audio_file or args.no_audio):
        raise SystemExit("--record-eval records the camera and the microphone, so it cannot be combined with "
                         "--replay, --audio-file or --no-audio")
    if args.replay:
        for f in args.replay:
            if not f.exists():
                raise SystemExit(f"--replay: {f} not found")
        if args.replay[0].suffix.lower() not in VIDEO_TYPES:
            raise SystemExit(f"--replay: the browser plays {sorted(VIDEO_TYPES)} files, not {args.replay[0].suffix}")
        load_audio_file(args.replay[1])  # fail now on a WAV that is not 16 kHz mono 16 bit
    SETTINGS.update(stats_dir=args.stats_dir, label=args.label, audio_file=args.audio_file, audio=not args.no_audio,
                    replay=tuple(str(f.resolve()) for f in args.replay) if args.replay else None,
                    data_dir=args.data_dir, record_eval=args.record_eval)
    if args.record_eval:
        print(f"RECORDING FOR EVAL: answers are saved as video and audio to {recordings.RECORDINGS_DIR}", flush=True)
    # No keepalive pings: the socket is on localhost and the browser closes it with the tab.
    # With uvicorn's default (20 s ping, 20 s pong timeout) sessions closed at 40 s while
    # frames were streaming (2026-09-25 test), cause not found.
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws="wsproto", ws_ping_interval=None, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
