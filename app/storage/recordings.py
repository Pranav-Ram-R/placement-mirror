"""Eval recordings (development only, server flag --record-eval).

Raw camera video and microphone audio of each answer, for evaluation, saved to
eval/recordings/<session id>/ (gitignored):
- audio.wav: 16 kHz mono 16 bit, the same sounddevice blocks the audio pipeline gets
- video.webm: the browser's MediaRecorder recording of the same getUserMedia stream
- recording.json: session id, question id, and per stream the start time (Unix epoch
  seconds, this machine's wall clock) and the duration measured from the file

Both streams start with the answer and stop on the answer stop event. The microphone
delivers its first samples some time after its stream starts (0.27 s on the development
laptop, MME host API), so the browser starts its recorder when the first audio block has
arrived (the "eval_recording" "started" event), not on the answering state. The normal app
never writes raw media: nothing here runs unless the server was started with --record-eval.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np

RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "eval" / "recordings"
META = "recording.json"
_meta_lock = threading.Lock()


def update_meta(folder: Path, **fields) -> dict:
    """Merge fields into the folder's recording.json (audio and video are written separately)."""
    path = Path(folder) / META
    with _meta_lock:
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        meta.update(fields)
        path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


class WavRecorder:
    """Writes captured audio blocks to a 16 bit mono WAV as they arrive."""

    def __init__(self, path: Path, sample_rate: int = 16000, on_start=None):
        self.path = Path(path)
        self.rate = sample_rate
        self.on_start = on_start  # called once with start_epoch_s when the first block is written
        self.samples = 0
        self.start_epoch_s: float | None = None
        self._lock = threading.Lock()
        self._wav = wave.open(str(self.path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)

    def write(self, block) -> None:
        """block: app.audio.vad.Block, stamped when the block was complete."""
        first = False
        with self._lock:
            if self._wav is None:
                return
            if self.start_epoch_s is None:  # the first sample is one block before the stamp
                self.start_epoch_s = time.time() - (time.monotonic() - block.t_mono) - len(block.samples) / self.rate
                first = True
            pcm = (np.clip(block.samples, -1.0, 1.0) * 32767.0).astype("<i2")
            self._wav.writeframes(pcm.tobytes())
            self.samples += len(pcm)
        if first and self.on_start is not None:
            self.on_start(self.start_epoch_s)

    def close(self) -> dict:
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None
        return {"file": self.path.name, "start_epoch_s": self.start_epoch_s, "sample_rate": self.rate,
                "samples": self.samples, "duration_s": round(self.samples / self.rate, 4)}


# ------------------------------------------------------------------ WebM (Matroska) timing

_MASTERS = {0x18538067, 0x1F43B675, 0x1549A966, 0x1654AE6B, 0xAE, 0xA0}  # Segment Cluster Info Tracks TrackEntry BlockGroup
_EBML_HEADER, _TIMECODE_SCALE, _CLUSTER_TIMECODE = 0x1A45DFA3, 0x2AD7B1, 0xE7
_SIMPLE_BLOCK, _BLOCK, _TRACK_NUMBER, _TRACK_TYPE = 0xA3, 0xA1, 0xD7, 0x83


def _vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int, bool]:
    first = data[pos]
    length = 1
    while length <= 8 and not first & (0x80 >> (length - 1)):
        length += 1
    if length > 8:
        raise ValueError(f"bad EBML variable length integer at byte {pos}")
    value = first if keep_marker else first & (0xFF >> length)
    for b in data[pos + 1:pos + length]:
        value = (value << 8) | b
    unknown = not keep_marker and value == (1 << (7 * length)) - 1
    return value, pos + length, unknown


def webm_frame_times(data: bytes) -> list[float]:
    """Presentation time in seconds of every video block, in file order.

    Walks the EBML elements flat: master elements are entered, others skipped. MediaRecorder
    writes the Segment and Clusters with unknown sizes, which a flat walk handles.
    """
    scale_ns, cluster_tc, video_tracks, times = 1_000_000, 0, set(), []
    track: dict = {}
    pos = 0
    while pos < len(data):
        try:
            eid, p, _ = _vint(data, pos, keep_marker=True)
            size, p, unknown = _vint(data, p, keep_marker=False)
        except (ValueError, IndexError):
            break  # a truncated last element
        if eid in _MASTERS:
            if eid == 0xAE:
                track = {}
            pos = p
            continue
        if unknown or p + size > len(data):
            break
        body = data[p:p + size]
        if eid == _EBML_HEADER:
            pass
        elif eid == _TIMECODE_SCALE:
            scale_ns = int.from_bytes(body, "big")
        elif eid == _CLUSTER_TIMECODE:
            cluster_tc = int.from_bytes(body, "big")
        elif eid in (_TRACK_NUMBER, _TRACK_TYPE):
            track[eid] = int.from_bytes(body, "big")
            if track.get(_TRACK_TYPE) == 1 and _TRACK_NUMBER in track:
                video_tracks.add(track[_TRACK_NUMBER])
        elif eid in (_SIMPLE_BLOCK, _BLOCK):
            number, q, _ = _vint(body, 0, keep_marker=False)
            if not video_tracks or number in video_tracks:
                rel = int.from_bytes(body[q:q + 2], "big", signed=True)
                times.append((cluster_tc + rel) * scale_ns / 1e9)
        pos = p + size
    return times


def webm_timing(path: Path) -> dict:
    """Frame count and duration: last frame time - first frame time + the median frame interval."""
    times = sorted(webm_frame_times(Path(path).read_bytes()))
    if not times:
        return {"frames": 0, "duration_s": None}
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return {"frames": len(times), "first_frame_s": round(times[0], 4), "last_frame_s": round(times[-1], 4),
            "median_frame_interval_s": round(step, 4), "duration_s": round(times[-1] - times[0] + step, 4),
            "duration_formula": "last_frame_s - first_frame_s + median_frame_interval_s"}


def _compare(folder: Path) -> dict:
    """Once both streams are saved, add their duration difference and start offset."""
    meta = update_meta(folder)
    video, audio = meta.get("video") or {}, meta.get("audio") or {}
    if video.get("duration_s") is None or audio.get("duration_s") is None:
        return meta
    diff = {"duration_difference_ms": round((video["duration_s"] - audio["duration_s"]) * 1000, 1),
            "duration_difference_formula": "(video.duration_s - audio.duration_s) * 1000"}
    if video.get("start_epoch_s") and audio.get("start_epoch_s"):
        diff["video_start_minus_audio_start_ms"] = round((video["start_epoch_s"] - audio["start_epoch_s"]) * 1000, 1)
    return update_meta(folder, **diff)


def save_audio(folder: Path, audio: dict) -> dict:
    update_meta(folder, audio=audio)
    return _compare(folder)


def save_video(folder: Path, data: bytes, *, start_epoch_ms: float | None, stop_epoch_ms: float | None,
               mime: str) -> dict:
    """Store the browser's WebM next to the WAV and add its timing to recording.json."""
    folder = Path(folder)
    path = folder / "video.webm"
    path.write_bytes(data)
    video = {"file": path.name, "mime_type": mime, "bytes": len(data),
             "start_epoch_s": start_epoch_ms / 1000 if start_epoch_ms else None,
             "stop_epoch_s": stop_epoch_ms / 1000 if stop_epoch_ms else None, **webm_timing(path)}
    update_meta(folder, video=video)
    return _compare(folder)


def start_folder(root: Path, session_id: str, question: dict) -> Path:
    folder = Path(root) / session_id
    folder.mkdir(parents=True, exist_ok=True)
    update_meta(folder, session_id=session_id, question_id=question["id"], question=question["text"],
                created=dt.datetime.now().isoformat(timespec="seconds"),
                clock="start_epoch_s values are Unix epoch seconds from this machine's wall clock")
    return folder


def list_recordings(root: Path = RECORDINGS_DIR) -> list[dict]:
    """Every recording folder with a recording.json, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if (p / META).exists()):
        meta = json.loads((d / META).read_text(encoding="utf-8"))
        out.append({"id": d.name, "dir": d, "meta": meta,
                    "audio": d / "audio.wav" if (d / "audio.wav").exists() else None,
                    "video": d / "video.webm" if (d / "video.webm").exists() else None})
    return out
