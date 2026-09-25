"""The answer report, computed from the saved timeline (app.session.controller).

Definitions (times are seconds from answer start):
- Face per frame, three states: facing, not_facing (face found, turned away) and
  not_visible (no face found). Facing camera % = facing frames / all frames. Not facing %
  and face not visible % are reported separately, over all frames too.
- Slouching % and leaning %: frames with the flag set / frames with a posture result.
  Frames without one (shoulders not seen, or posture not calibrated) are left out.
- WPM: words / time * 60. Overall uses the whole answer length. A segment's words are
  spread evenly over the segment, as in the live indicator (app.analysis.live).
- Fillers: um, umm, uh, uhh, hmm and mm (app.analysis.live.FILLER_RE). A filler's time
  is estimated from its word position in the segment, Whisper gives no word times.
- Long pauses: VAD pause events of at least the timeline's pause_min_s. A silence still
  running when the answer stopped is listed as ending_silence and not counted.
- Windows: window_s long. A last window shorter than min_last_window_s is merged into
  the window before it.
"""

from __future__ import annotations

import re

import numpy as np

from app.analysis.feedback import build_feedback, content_feedback, guideline_dict
from app.analysis.live import FILLER_RE, WORD_RE
from app.config import REPORT, ReportConfig

# v2: face states (facing, not facing, face not visible). A saved v1 report is rebuilt from
# its timeline when read (app.storage.history).
REPORT_SCHEMA = "placement-mirror report v2"
FACE_STATES = ("facing", "not_facing", "not_visible")


def face_state(facing: bool | None) -> str:
    """The vision pipeline's facing flag as a face state. None means no face was found."""
    return "facing" if facing is True else "not_facing" if facing is False else "not_visible"


def frame_face(frame: dict) -> str:
    # v2 timelines carry face. v1 timelines carry facing: true, false or null (no face).
    return frame["face"] if "face" in frame else face_state(frame.get("facing"))


def pct(part: int, whole: int) -> float | None:
    return round(100.0 * part / whole, 2) if whole else None


def per_min(count: float, seconds: float) -> float | None:
    return round(count / seconds * 60.0, 2) if seconds > 0 else None


def make_windows(duration: float, cfg: ReportConfig = REPORT) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    starts = np.arange(0.0, duration, cfg.window_s).tolist()
    windows = [(s, min(s + cfg.window_s, duration)) for s in starts]
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < cfg.min_last_window_s:
        windows[-2:] = [(windows[-2][0], duration)]
    return [(round(a, 4), round(b, 4)) for a, b in windows]


def window_index(t: float, windows: list[tuple[float, float]]) -> int:
    for i, (_, end) in enumerate(windows):
        if t < end:
            return i
    return len(windows) - 1  # at or after the end of the answer


def word_times(segment: dict) -> list[float]:
    n = len(WORD_RE.findall(segment["text"]))
    return np.linspace(segment["start"], segment["end"], n + 2)[1:-1].tolist() if n else []


CONTEXT_WORDS = 5  # words of transcript shown around a filler or before a pause
BEFORE_RE = re.compile(r"(?:\S+\s+){0,%d}$" % CONTEXT_WORDS)
AFTER_RE = re.compile(r"\S*(?:\s+\S+){0,%d}" % CONTEXT_WORDS)


def locate_fillers(segments: list[dict]) -> list[dict]:
    fillers = []
    for seg in segments:
        text, times = seg["text"], word_times(seg)
        for m in FILLER_RE.finditer(text):
            word_pos = len(WORD_RE.findall(text[:m.start()]))
            fillers.append({"word": m.group(0), "segment": seg["index"], "char_start": m.start(),
                            "char_end": m.end(), "t": round(times[word_pos], 3),
                            # the text around the filler, spacing and punctuation kept
                            "before": BEFORE_RE.search(text[:m.start()]).group(0),
                            "after": AFTER_RE.match(text[m.end():]).group(0)})
    return fillers


def split_fillers(text: str) -> list[dict]:
    """Segment text as parts, with each filler as its own part, for highlighting."""
    parts, pos = [], 0
    for m in FILLER_RE.finditer(text):
        if m.start() > pos:
            parts.append({"text": text[pos:m.start()], "filler": False})
        parts.append({"text": m.group(0), "filler": True})
        pos = m.end()
    if pos < len(text):
        parts.append({"text": text[pos:], "filler": False})
    return parts


def build_report(timeline: dict, cfg: ReportConfig = REPORT) -> dict:
    duration = timeline["answer"]["duration_s"]
    frames, segments = timeline["frames"], sorted(timeline["segments"], key=lambda s: s["start"])
    windows = make_windows(duration, cfg)
    has_audio = timeline["audio_source"] is not None

    # vision, per window and overall
    rows = [{"start": a, "end": b, "frames": 0, **{s: 0 for s in FACE_STATES}, "posture_frames": 0, "slouching": 0,
             "leaning": 0, "words": 0} for a, b in windows]
    for f in frames:
        if not rows:
            break
        r = rows[window_index(f["t"], windows)]
        r["frames"] += 1
        r[frame_face(f)] += 1
        if f["slouching"] is not None:
            r["posture_frames"] += 1
            r["slouching"] += f["slouching"] is True
            r["leaning"] += f["leaning"] is True

    # speech, per window and overall
    for seg in segments:
        for t in word_times(seg):
            if rows:
                rows[window_index(t, windows)]["words"] += 1
    words = sum(len(WORD_RE.findall(s["text"])) for s in segments)
    fillers = locate_fillers(segments)
    pause_min = timeline["vad"]["pause_min_s"]
    long_pauses, ending_silence = [], None
    for p in sorted(timeline["pauses"], key=lambda p: p["start"]):
        before = [s for s in segments if s["start"] < p["start"]]
        item = {"start": p["start"], "end": p["end"], "duration_s": p["duration_s"],
                "after_segment": before[-1]["index"] if before else None,
                "after_text": " ".join(before[-1]["text"].split()[-CONTEXT_WORDS:]) if before else ""}
        if p.get("open_at_stop"):
            ending_silence = item
        elif p["duration_s"] >= pause_min:
            long_pauses.append(item)

    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    overall = {
        "duration_s": duration,
        "frames": len(frames),
        "facing_camera_pct": pct(total("facing"), len(frames)),
        "not_facing_pct": pct(total("not_facing"), len(frames)),
        "face_not_visible_pct": pct(total("not_visible"), len(frames)),
        "posture_frames": total("posture_frames"),
        "slouching_pct": pct(total("slouching"), total("posture_frames")),
        "leaning_pct": pct(total("leaning"), total("posture_frames")),
        "words": words if has_audio else None,
        "wpm": per_min(words, duration) if has_audio else None,
        "filler_count": len(fillers) if has_audio else None,
        "fillers_per_min": per_min(len(fillers), duration) if has_audio else None,
        "long_pause_count": len(long_pauses) if has_audio else None,
        "longest_pause_s": max((p["duration_s"] for p in long_pauses), default=None),
        "long_pauses_per_min": per_min(len(long_pauses), duration) if has_audio else None,
    }
    window_rows = [{"start": r["start"], "end": r["end"], "frames": r["frames"],
                    "facing_camera_pct": pct(r["facing"], r["frames"]),
                    "not_facing_pct": pct(r["not_facing"], r["frames"]),
                    "face_not_visible_pct": pct(r["not_visible"], r["frames"]), "posture_frames": r["posture_frames"],
                    "slouching_pct": pct(r["slouching"], r["posture_frames"]),
                    "leaning_pct": pct(r["leaning"], r["posture_frames"]), "words": r["words"],
                    "wpm": per_min(r["words"], r["end"] - r["start"]) if has_audio else None}
                   for r in rows]

    transcript = []
    for seg in segments:
        transcript.append({"kind": "speech", "segment": seg["index"], "start": seg["start"], "end": seg["end"],
                           "parts": split_fillers(seg["text"].strip())})
    for p in long_pauses + ([ending_silence] if ending_silence else []):
        transcript.append({"kind": "pause", "start": p["start"], "duration_s": p["duration_s"],
                           "at_end": p is ending_silence})
    transcript.sort(key=lambda item: item["start"])

    report = {
        "schema": REPORT_SCHEMA,
        "session_id": timeline["session_id"],
        "question": timeline["question"],
        "answer": timeline["answer"],
        "mode": timeline["mode"],
        "replay": timeline["replay"],
        "audio_source": timeline["audio_source"],
        "posture_calibrated": bool(timeline["calibration"].get("pose")),
        "window_s": cfg.window_s,
        "pause_min_s": pause_min,
        "overall": overall,
        "windows": window_rows,
        "fillers": fillers,
        "long_pauses": long_pauses,
        "ending_silence": ending_silence,
        "transcript": transcript,
    }
    report["guidelines"] = [guideline_dict(g) for g in cfg.guidelines]
    report["feedback"] = build_feedback(overall, cfg)
    report["content_feedback"] = content_feedback(timeline, report)
    return report
