"""Session history: one folder per answer under %LOCALAPPDATA%\\PlacementMirror\\sessions.

Each folder holds timeline.json (app.session.controller) and report.json
(app.analysis.report). A folder with a timeline but no report, for example from before
reports existed, gets its report built from the timeline when it is first read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.analysis.report import REPORT_SCHEMA, build_report
from app.storage import paths

SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
# The metrics the history view shows as trends, in order.
TREND_METRICS = [
    {"metric": "facing_camera_pct", "label": "Facing the camera", "unit": "%"},
    {"metric": "wpm", "label": "Speaking pace", "unit": "words per minute"},
    {"metric": "fillers_per_min", "label": "Filler words", "unit": "per minute"},
    {"metric": "long_pause_count", "label": "Long pauses", "unit": "count"},
]


def folder(session_id: str, sessions_dir: Path | None = None) -> Path:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise KeyError(f"not a session id: {session_id}")
    return (Path(sessions_dir) if sessions_dir else paths.sessions_dir()) / session_id


def save_report(report: dict, sessions_dir: Path | None = None) -> Path:
    path = folder(report["session_id"], sessions_dir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return path


def load_report(session_id: str, sessions_dir: Path | None = None) -> dict:
    d = folder(session_id, sessions_dir)
    report_path, timeline_path = d / "report.json", d / "timeline.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") == REPORT_SCHEMA:
            return report
    if not timeline_path.exists():
        raise KeyError(f"no session {session_id}")
    report = build_report(json.loads(timeline_path.read_text(encoding="utf-8")))
    save_report(report, sessions_dir)
    return report


def list_sessions(sessions_dir: Path | None = None) -> list[dict]:
    """Every saved answer, oldest first, with the trend metrics."""
    root = Path(sessions_dir) if sessions_dir else paths.sessions_dir()
    if not root.is_dir():
        return []
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and SESSION_ID_RE.fullmatch(p.name)):
        try:
            report = load_report(d.name, root)
        except (KeyError, ValueError, TypeError, OSError) as e:
            rows.append({"session_id": d.name, "error": f"{type(e).__name__}: {e}"})
            continue
        overall = report["overall"]
        rows.append({"session_id": report["session_id"], "started": report["answer"]["started"],
                     "question": report["question"], "duration_s": overall["duration_s"],
                     "replay": report["replay"] is not None,
                     **{m["metric"]: overall[m["metric"]] for m in TREND_METRICS}})
    return rows
