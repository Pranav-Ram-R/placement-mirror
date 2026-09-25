"""Rule-based feedback: three improvements from the report metrics.

Each measured metric is compared with its coaching guideline (config.ReportConfig). The
guidelines are coaching targets, not measurements. Distance from a guideline is relative
to the limit it passes:
    upper limit: (value - high) / high
    lower limit: (low - value) / low
Positive means outside the guideline. For a range the larger of the two is used, so a
value inside the range gets a negative distance that is closest to zero near its nearer
edge. Metrics are ranked by distance, largest first, and the top three are returned,
also when some of them are inside their guideline.

Feedback on what was said (for example a STAR check on behavioral answers) is out of
scope for now. content_feedback() is the hook for the local LLM.
"""

from __future__ import annotations

from typing import Callable

from app.config import REPORT, Guideline, ReportConfig

IMPROVEMENTS = 3

# One concrete action per metric and direction ("high": above the upper limit or near it,
# "low": below the lower limit or near it).
ACTIONS = {
    ("facing_camera_pct", "low"): "Put the interview window right below the camera and look at the lens at the "
                                  "start and end of each point.",
    ("slouching_pct", "high"): "Sit back against the chair with both feet flat and raise the screen so your head "
                               "stays level.",
    ("leaning_pct", "high"): "Keep both shoulders level. Rest both forearms on the desk instead of leaning on one "
                             "elbow.",
    ("wpm", "high"): "Slow down. Take one breath at the end of each sentence before you start the next.",
    ("wpm", "low"): "Pick two or three points before you start, so you can move from one to the next without "
                    "searching for words.",
    ("fillers_per_min", "high"): "When you need time to think, stay silent for a moment instead of saying um or "
                                 "uh.",
    ("long_pauses_per_min", "high"): "Before you answer, list two or three points in your head, and go straight "
                                     "to the next one when a point is done.",
}


def distance(value: float, g: Guideline) -> tuple[float, str]:
    """Signed relative distance from the guideline, and the direction it leans."""
    options = []
    if g.high is not None:
        options.append(((value - g.high) / g.high, "high"))
    if g.low is not None:
        options.append(((g.low - value) / g.low, "low"))
    return max(options)


def guideline_text(g: Guideline) -> str:
    unit = g.unit if g.unit.startswith("%") else f" {g.unit}"
    if g.low is not None and g.high is not None:
        return f"{g.low:g} to {g.high:g}{unit}"
    if g.low is not None:
        return f"at least {g.low:g}{unit}"
    return f"at most {g.high:g}{unit}"


def guideline_dict(g: Guideline) -> dict:
    return {"metric": g.metric, "label": g.label, "unit": g.unit, "low": g.low, "high": g.high, "kind": g.kind,
            "text": guideline_text(g)}


def build_feedback(overall: dict, cfg: ReportConfig = REPORT) -> dict:
    ranked = []
    for g in cfg.guidelines:
        value = overall.get(g.metric)
        if value is None:
            continue  # not measured in this answer
        d, direction = distance(value, g)
        ranked.append({
            "metric": g.metric, "label": g.label, "value": value, "unit": g.unit,
            "guideline": guideline_dict(g),
            "outside_guideline": d > 0, "distance": round(d, 4), "action": ACTIONS[(g.metric, direction)],
        })
    ranked.sort(key=lambda item: item["distance"], reverse=True)
    note = None
    if len(ranked) < IMPROVEMENTS:
        note = f"This answer measured {len(ranked)} of the {len(cfg.guidelines)} metrics, so the list has fewer " \
               f"than {IMPROVEMENTS} items."
    return {"improvements": ranked[:IMPROVEMENTS], "ranked": ranked, "note": note,
            "distance_formula": "upper limit: (value - high) / high, lower limit: (low - value) / low. "
                                "Positive means outside the guideline."}


# Hook for LLM content feedback on the transcript, for example a STAR check on behavioral
# answers: a callable (timeline, report) -> dict. Nothing is registered yet.
CONTENT_FEEDBACK: Callable[[dict, dict], dict] | None = None


def content_feedback(timeline: dict, report: dict) -> dict | None:
    return CONTENT_FEEDBACK(timeline, report) if CONTENT_FEEDBACK is not None else None
