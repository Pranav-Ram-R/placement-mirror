"""Local web server: serves the UI and runs one vision pipeline per WebSocket session.

- GET /             the UI in app/ui/static/
- GET /api/status   ModelRunner.status(): the real compute unit and load path per model
- WS  /ws           one session. Binary messages carry frames in (app.vision.pipeline
                    parse_frame). Text messages carry JSON commands in ({"type":
                    "calibrate"} or {"type": "stop"}) and JSON events out.

Models load once at startup through ModelRunner. Binds to 127.0.0.1 only and makes no
network calls. One session at a time, because all sessions would share the same models
and CPU.

When a session ends (Stop in the UI or the socket closes) the per stage p50/p95 table is
printed. With --stats-dir the session report and the stage percentiles are saved there as
Measurement records.

Usage: python -m app.server [--port 8000] [--stats-dir DIR] [--label "what the input was"]
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

from app.runtime.runner import CPU_ONLY, ModelRunner
from app.vision.pipeline import VISION_MODELS, VisionPipeline

STATIC = Path(__file__).resolve().parent / "ui" / "static"
SETTINGS = {"stats_dir": None, "label": ""}
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
    runner = ModelRunner()
    for name in VISION_MODELS:
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


def print_report(report: dict) -> None:
    print(f"\nSession: {report['duration_s']:.1f} s, frames received {report['frames_received']}, "
          f"processed {report['frames_processed']}, replaced in mailbox {report['frames_replaced_in_mailbox']}, "
          f"dropped in browser {report.get('browser', {}).get('dropped_busy', 'n/a')}, "
          f"face detector runs {report['face_detector_runs']}")
    print("| stage | count | p50 ms | p95 ms |")
    print("|---|---|---|---|")
    for stage, s in report["stages"].items():
        print(f"| {stage} | {s['count']} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} |")
    sys.stdout.flush()


def save_report(report: dict, status: dict, stats_dir: Path) -> Path:
    from benchmarks.schema import Measurement, Source, save_json

    src = Source.LOCAL_X86_CPU if platform.machine().upper() in ("AMD64", "X86_64") else Source.PHYSICAL_SNAPDRAGON
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stats_dir.mkdir(parents=True, exist_ok=True)
    units = {n: s["compute_unit"] for n, s in status.items()}
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
                exec_precision=None,
                ort_version=STATE["runner"].ort.__version__,
                notes=f"{s['count']} samples, numpy.percentile linear. {session}"))
    path = stats_dir / f"{stamp}_vision_session.json"
    save_json(records, path)
    (stats_dir / f"{stamp}_vision_session_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def finish(pipeline: VisionPipeline, browser: dict) -> dict:
    report = pipeline.report()
    report["browser"] = browser
    if report["frames_processed"]:
        print_report(report)
        if SETTINGS["stats_dir"]:
            path = save_report(report, STATE["runner"].status(), Path(SETTINGS["stats_dir"]))
            print(f"saved stage measurements to {path}", flush=True)
    return report


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

    pipeline = VisionPipeline(STATE["runner"], emit)
    await ws.send_json({"type": "status", **status_payload()})
    pipeline.start()
    sender = asyncio.create_task(send_events())
    browser: dict = {}
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                print(f"session closed by the client, code {msg.get('code')} {msg.get('reason') or ''}", flush=True)
                break
            if msg.get("bytes") is not None:
                pipeline.submit(msg["bytes"], time.monotonic(), time.perf_counter())
            elif msg.get("text") is not None:
                cmd = json.loads(msg["text"])
                if cmd.get("type") == "calibrate":
                    pipeline.calibrate(time.monotonic())
                elif cmd.get("type") == "stop":
                    browser = cmd.get("browser", {})
                    pipeline.stop()
                    report = finish(pipeline, browser)
                    emit({"type": "stats", **report})
                    pipeline = None
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if pipeline is not None:
            pipeline.stop()
            finish(pipeline, browser)
        await asyncio.sleep(0.05)  # let queued events go out
        sender.cancel()
        lock.release()


app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stats-dir", type=Path, help="save each session's stage percentiles here as Measurements")
    ap.add_argument("--label", default="", help="what the session input was, saved in the Measurement notes")
    args = ap.parse_args()
    SETTINGS["stats_dir"], SETTINGS["label"] = args.stats_dir, args.label
    # No keepalive pings: the socket is on localhost and the browser closes it with the tab.
    # With uvicorn's default (20 s ping, 20 s pong timeout) sessions closed at 40 s while
    # frames were streaming (2026-09-25 test), cause not found.
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws="wsproto", ws_ping_interval=None, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
