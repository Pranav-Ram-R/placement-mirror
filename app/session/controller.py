"""One interview session: the state machine driving the vision and audio pipelines.

Vision runs for the whole WebSocket session (preview indicators and calibration). Audio
runs only while the user answers: the audio pipeline starts on "start answer" and is
drained on stop, so the last segment is transcribed before the timeline is built.

The saved timeline (sessions_dir/<session id>/timeline.json) covers the answer only:
- frames: every processed video frame between start and stop, with time from answer
  start, face (facing, not_facing or not_visible), yaw, pitch, slouching and leaning
- segments: every speech segment, with start, end, text, fillers and word count
- pauses: pause intervals from the VAD pause events
The report (report.json in the same folder, app.analysis.report) is built from it.
Only metrics and transcript text are written. Raw video and audio never are, except in
eval recording mode (eval_recordings set, server flag --record-eval, development only):
then the microphone audio of each answer goes to eval_recordings/<session id>/audio.wav
and the browser uploads its camera video there (app.storage.recordings).
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Callable

from app.analysis.report import build_report, face_state
from app.config import AUDIO, SESSION, VISION, SessionConfig, VisionConfig
from app.runtime.priority import AsrPriority
from app.session.machine import InvalidTransition, SessionMachine, State
from app.session.questions import default_bank, find_question
from app.storage import history, paths, recordings
from app.vision.pipeline import VisionPipeline

# v2: frames carry face ("facing", "not_facing" or "not_visible") instead of v1's facing
# (true, false or null).
TIMELINE_SCHEMA = "placement-mirror timeline v2"


class SessionController:
    def __init__(self, runner, emit: Callable[[dict], None], *, bank: dict | None = None,
                 sessions_dir: Path | None = None, audio_factory=None, replay: dict | None = None,
                 cfg: SessionConfig = SESSION, vision_cfg: VisionConfig = VISION,
                 eval_recordings: Path | None = None):
        self.runner = runner
        self.emit = emit
        self.bank = bank or default_bank()
        self.sessions_dir = Path(sessions_dir) if sessions_dir else paths.sessions_dir()
        self.audio_factory = audio_factory  # (emit, priority, recorder) -> AudioPipeline, or None for no audio
        self.eval_recordings = Path(eval_recordings) if eval_recordings else None  # None: never record media
        self.recorder: recordings.WavRecorder | None = None
        self.record_folder: Path | None = None
        self.replay = replay
        self.cfg = cfg
        self.priority = AsrPriority()
        self.machine = SessionMachine()
        self.vision = VisionPipeline(runner, self._vision_event, vision_cfg, priority=self.priority)
        self.audio = None
        self.audio_runs: list = []  # every audio pipeline used in this connection, for the timing report
        self.answer: dict | None = None
        self.calibration_event: dict | None = None
        self.last_timeline: Path | None = None
        self.last_report: Path | None = None
        self._timer: threading.Timer | None = None
        self._processing: threading.Thread | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ events out

    def state_event(self, **extra) -> dict:
        q = self.machine.question
        return {"type": "session_state", "state": self.machine.state.value, "question": q,
                "limit_s": (q["suggested_time_s"] + self.cfg.auto_stop_extra_s) if q else None, **extra}

    def _emit_state(self, **extra) -> None:
        self.emit(self.state_event(**extra))

    def _vision_event(self, ev: dict) -> None:
        self.emit(ev)
        if ev.get("type") != "calibration" or ev.get("state") not in ("done", "partial", "failed"):
            return
        with self._lock:
            if self.machine.state != State.CALIBRATE:
                return
            self.calibration_event = ev
            if ev["face"] or not self.cfg.require_face_calibration:
                self.machine.fire("calibrated")
                note = "" if ev["pose"] else "Shoulders were not seen, so posture will not be measured."
                self._emit_state(message=note)
            else:
                self._emit_state(message=f"Calibration did not see a face. {ev.get('message', '')} Try again.")

    # ------------------------------------------------------------------ commands in

    def start(self) -> None:
        self.vision.start()
        self._emit_state()

    def command(self, cmd: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        kind = cmd.get("type")
        try:
            with self._lock:
                if kind == "select_question":
                    self.machine.fire("select", find_question(self.bank, cmd.get("id", "")))
                    self._emit_state()
                elif kind == "calibrate":
                    if self.machine.state in (State.READY, State.REPORT):
                        self.machine.fire("recalibrate")
                    if self.machine.state != State.CALIBRATE:
                        raise InvalidTransition("choose a question before calibrating")
                    self.vision.calibrate(now)
                    self._emit_state(message="Calibrating. Face the camera and sit straight for 3 seconds.")
                elif kind == "start_answer":
                    self.start_answer(now)
                elif kind == "stop_answer":
                    # the browser sends reason "end of replay video" when a replay file ends
                    reason = "end of replay video" if cmd.get("reason") == "end of replay video" else "user"
                    self.stop_answer(reason, now)
                else:
                    raise InvalidTransition(f"unknown command {kind}")
        except (InvalidTransition, KeyError) as e:
            self.emit({"type": "error", "message": str(e).strip("'")})

    def start_answer(self, now: float) -> None:
        with self._lock:
            self.machine.fire("start")
            q = self.machine.question
            limit = q["suggested_time_s"] + self.cfg.auto_stop_extra_s
            started = dt.datetime.now()
            session_id = f"{started.strftime('%Y-%m-%d_%H%M%S')}_{q['id']}"
            self.answer = {"start_mono": now, "start_wall": started.isoformat(timespec="seconds"),
                           "session_id": session_id, "limit_s": limit, "stopped_by": None}
            self.audio = None
            self.recorder = self.record_folder = None
            if self.eval_recordings is not None and self.audio_factory is not None:
                self.record_folder = recordings.start_folder(self.eval_recordings, session_id, q)
                self.recorder = recordings.WavRecorder(
                    self.record_folder / "audio.wav", AUDIO.sample_rate,
                    on_start=lambda t: self.emit({"type": "eval_recording", "state": "started",
                                                  "session_id": session_id, "audio_start_epoch_s": t}))
            if self.audio_factory is not None:
                try:
                    self.audio = self.audio_factory(self.emit, self.priority, self.recorder)
                    self.audio.start()
                    self.audio_runs.append(self.audio)
                    self.emit({"type": "audio", "state": "running", "source": self.audio.source.describe()})
                except Exception as e:  # noqa: BLE001  the answer goes on without audio
                    self.audio = None
                    self.emit({"type": "error", "message": f"Audio could not start: {type(e).__name__}: {e}"})
            self._timer = threading.Timer(limit, self.stop_answer, args=(f"auto stop at {limit:g} s",))
            self._timer.daemon = True
            self._timer.start()
            self._emit_state(session_id=session_id, recording=self.recorder is not None)

    def stop_answer(self, reason: str, now: float | None = None, wait: bool = False) -> None:
        with self._lock:
            if self.machine.state != State.ANSWERING:
                return  # already stopped (for example the timer after a user stop)
            now = time.monotonic() if now is None else now
            self.machine.fire("stop")
            if self._timer is not None:
                self._timer.cancel()
            self.answer.update(end_mono=now, stopped_by=reason)
            self._emit_state(message=f"Answer stopped ({reason}). Processing.")
            self._processing = threading.Thread(target=self._process, name="session-processing", daemon=True)
            self._processing.start()
        if wait:
            self.wait_processed()

    def wait_processed(self, timeout: float | None = None) -> None:
        if self._processing is not None:
            self._processing.join(timeout)

    def _process(self) -> None:
        if self.audio is not None:
            self.audio.stop()
        if self.recorder is not None:
            try:
                source = self.audio.source.describe() if self.audio is not None else None
                meta = recordings.save_audio(self.record_folder, {**self.recorder.close(), "source": source})
                self.emit({"type": "eval_recording", "state": "audio_saved", "session_id": meta["session_id"],
                           "audio": meta["audio"]})
            except Exception as e:  # noqa: BLE001  the answer is still processed
                self.emit({"type": "error", "message": f"Eval audio recording failed: {type(e).__name__}: {e}"})
        try:
            timeline = self.build_timeline()
            path = self.save_timeline(timeline)
            report_path = history.save_report(build_report(timeline), self.sessions_dir)
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "error", "message": f"Processing failed: {type(e).__name__}: {e}"})
            return
        with self._lock:
            self.machine.fire("processed")
            self.last_timeline, self.last_report = path, report_path
            self._emit_state(timeline=str(path), summary=timeline["summary"], report_id=timeline["session_id"])

    def close(self) -> None:
        """Connection closed: finish an answer in progress, then stop the vision worker."""
        self.stop_answer("connection closed", wait=True)
        self.wait_processed()
        if self._timer is not None:
            self._timer.cancel()
        self.vision.stop()

    # ------------------------------------------------------------------ timeline

    def build_timeline(self) -> dict:
        a = self.answer
        start, end = a["start_mono"], a["end_mono"]
        rel = lambda t: round(t - start, 4)  # noqa: E731
        frames = [{"t": rel(f["t"]), "frame_id": f["frame_id"], "face_state": f["face_state"], "face": face_state(f["facing"]),
                   "yaw": f["yaw"], "pitch": f["pitch"], "slouching": f["slouching"], "leaning": f["leaning"],
                   "pose_updated": f["pose_updated"]}
                  for f in list(self.vision.frame_log) if start <= f["t"] <= end]
        segments, pauses, audio_source = [], [], None
        if self.audio is not None:
            audio_source = self.audio.source.describe()
            for r in sorted(self.audio.records, key=lambda r: r["start_mono"]):
                segments.append({"index": r["index"], "start": rel(r["start_mono"]), "end": rel(r["end_mono"]),
                                 "text": r["text"], "fillers": r["fillers"], "word_count": r["word_count"],
                                 "closed_by": r["closed_by"]})
            open_pause = None
            for ev in self.audio.pauses:
                if ev["state"] == "started":
                    open_pause = {"start": rel(ev["since_mono"]), "end": None, "duration_s": None}
                elif ev["state"] == "ended" and open_pause is not None:
                    open_pause["duration_s"] = round(ev["duration_s"], 4)
                    open_pause["end"] = round(open_pause["start"] + ev["duration_s"], 4)
                    pauses.append(open_pause)
                    open_pause = None
            if open_pause is not None:  # still silent when the answer stopped
                open_pause.update(end=rel(end), duration_s=round(rel(end) - open_pause["start"], 4), open_at_stop=True)
                pauses.append(open_pause)
        cal = self.vision.calibration
        status = self.runner.status()
        duration = end - start
        return {
            "schema": TIMELINE_SCHEMA,
            "session_id": None,
            "question": self.machine.question,
            "answer": {"started": a["start_wall"], "duration_s": round(duration, 4), "limit_s": a["limit_s"],
                       "stopped_by": a["stopped_by"]},
            "mode": self.vision.mode,
            "replay": self.replay,
            "audio_source": audio_source,
            "calibration": {"state": cal.state, "face": cal.face, "pose": cal.pose, "message": cal.message},
            "thresholds": {k: getattr(self.vision.cfg, k) for k in (
                "facing_max_yaw_deg", "facing_max_pitch_deg", "slouch_max_head_drop", "lean_max_tilt_deg")},
            "vad": {k: getattr(AUDIO, k) for k in ("segment_end_silence_ms", "segment_max_s", "speech_pad_ms",
                                                   "pause_min_s")},
            "compute_units": {n: s["compute_unit"] for n, s in status.items()},
            "summary": {"frames": len(frames), "first_frame_t": frames[0]["t"] if frames else None,
                        "last_frame_t": frames[-1]["t"] if frames else None, "segments": len(segments),
                        "words": sum(s["word_count"] for s in segments), "pauses": len(pauses)},
            "frames": frames,
            "segments": segments,
            "pauses": pauses,
        }

    def save_timeline(self, timeline: dict) -> Path:
        session_id = self.answer["session_id"]  # set when the answer started
        timeline["session_id"] = session_id
        folder = self.sessions_dir / session_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "timeline.json"
        path.write_text(json.dumps(timeline, indent=1) + "\n", encoding="utf-8")
        return path
