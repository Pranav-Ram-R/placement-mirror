import json

import pytest

from app.analysis.feedback import build_feedback
from app.analysis.report import REPORT_SCHEMA, build_report, make_windows
from app.config import GUIDELINE_KIND
from app.storage import history

SEG0 = " Um, I think the answer is uh yes."
SEG1 = " Hmm, hardware design is fun and I like testing it."


def make_timeline(session_id="2026-09-25_120000_t-1", duration=25.0, audio=True, pose=True):
    """A 25 s answer with 50 frames, 0.5 s apart.

    face: facing for t < 10, every other frame facing for 10 <= t < 20, no face after 20
    posture (none before t = 5): slouching for 10 <= t < 15, leaning for t >= 22.5
    speech: 8 words (Um, uh) from 1 to 4 s, 10 words (Hmm) from 12 to 20 s
    pauses: 4 to 12 s, and a silence from 20 s still running at the stop
    """
    frames = []
    for i in range(50):
        t = i * 0.5
        facing = True if t < 10 else (i % 2 == 0) if t < 20 else None
        has_pose = pose and t >= 5
        frames.append({"t": t, "frame_id": i, "face_state": "no face" if facing is None else "tracking",
                       "facing": facing, "yaw": None if facing is None else 0.0,
                       "pitch": None if facing is None else 0.0, "pose_updated": True,
                       "slouching": (10 <= t < 15) if has_pose else None,
                       "leaning": (t >= 22.5) if has_pose else None})
    segments = [
        {"index": 0, "start": 1.0, "end": 4.0, "text": SEG0, "fillers": ["Um", "uh"], "word_count": 8,
         "closed_by": "silence"},
        {"index": 1, "start": 12.0, "end": 20.0, "text": SEG1, "fillers": ["Hmm"], "word_count": 10,
         "closed_by": "silence"},
    ] if audio else []
    pauses = [{"start": 4.0, "end": 12.0, "duration_s": 8.0},
              {"start": 20.0, "end": 25.0, "duration_s": 5.0, "open_at_stop": True}] if audio else []
    return {
        "schema": "placement-mirror timeline v1", "session_id": session_id,
        "question": {"id": "t-1", "category": "hr", "text": "Tell me about yourself.", "type": "behavioral",
                     "suggested_time_s": 30},
        "answer": {"started": "2026-09-25T12:00:00", "duration_s": duration, "limit_s": 90, "stopped_by": "user"},
        "mode": "cpu_fallback", "replay": None, "audio_source": "fake audio" if audio else None,
        "calibration": {"state": "done", "face": {"yaw": 0.0, "pitch": 0.0, "frames": 20},
                        "pose": {"tilt_deg": 0.0, "head_ratio": 0.4, "frames": 5} if pose else None, "message": ""},
        "thresholds": {}, "vad": {"segment_end_silence_ms": 1000, "segment_max_s": 20.0, "speech_pad_ms": 200,
                                  "pause_min_s": 2.0},
        "compute_units": {}, "summary": {}, "frames": frames, "segments": segments, "pauses": pauses,
    }


def load_via_json(tmp_path, timeline):
    """The report is built from the saved timeline JSON, as the app does."""
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(timeline), encoding="utf-8")
    return build_report(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- metrics

def test_windows_are_10_s_and_a_short_tail_is_merged():
    assert make_windows(25.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    assert make_windows(32.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 32.0)]  # 2 s tail merged
    assert make_windows(9.0) == [(0.0, 9.0)]
    assert make_windows(0.0) == []


def test_report_metrics_match_the_timeline(tmp_path):
    r = load_via_json(tmp_path, make_timeline())
    assert r["schema"] == REPORT_SCHEMA
    o = r["overall"]
    # facing: 20 + 10 of 50 frames. Frames without a face count as not facing.
    assert o["frames"] == 50 and o["facing_camera_pct"] == 60.0
    # posture: 40 frames with a posture result, 10 slouching, 5 leaning
    assert o["posture_frames"] == 40 and o["slouching_pct"] == 25.0 and o["leaning_pct"] == 12.5
    # speech: 18 words in 25 s
    assert o["words"] == 18 and o["wpm"] == pytest.approx(18 / 25 * 60)
    assert o["filler_count"] == 3 and o["fillers_per_min"] == pytest.approx(3 / 25 * 60)
    # one long pause. The silence running at the stop is not counted.
    assert o["long_pause_count"] == 1 and o["longest_pause_s"] == 8.0
    assert o["long_pauses_per_min"] == pytest.approx(1 / 25 * 60)

    w = r["windows"]
    assert [(x["start"], x["end"], x["frames"]) for x in w] == [(0.0, 10.0, 20), (10.0, 20.0, 20), (20.0, 25.0, 10)]
    assert [x["facing_camera_pct"] for x in w] == [100.0, 50.0, 0.0]
    assert [x["posture_frames"] for x in w] == [10, 20, 10]
    assert [x["slouching_pct"] for x in w] == [0.0, 50.0, 0.0]
    assert [x["leaning_pct"] for x in w] == [0.0, 0.0, 50.0]
    assert [x["words"] for x in w] == [8, 10, 0]
    assert [x["wpm"] for x in w] == [48.0, 60.0, 0.0]


def test_fillers_and_pauses_are_located_in_the_transcript(tmp_path):
    r = load_via_json(tmp_path, make_timeline())
    f = r["fillers"]
    assert [(x["word"], x["segment"]) for x in f] == [("Um", 0), ("uh", 0), ("Hmm", 1)]
    assert SEG0[f[1]["char_start"]:f[1]["char_end"]] == "uh"
    # word times spread over the segment: 8 words in 1..4 s, 10 words in 12..20 s
    assert f[0]["t"] == pytest.approx(1 + 3 / 9, abs=1e-3) and f[1]["t"] == pytest.approx(1 + 7 * 3 / 9, abs=1e-3)
    assert f[2]["t"] == pytest.approx(12 + 8 / 11, abs=1e-3)
    assert f[1]["before"] == "I think the answer is " and f[1]["after"] == " yes."
    assert f[0]["before"] == "" and f[0]["after"] == ", I think the answer is"
    assert r["long_pauses"] == [{"start": 4.0, "end": 12.0, "duration_s": 8.0, "after_segment": 0,
                                 "after_text": "the answer is uh yes."}]
    assert r["ending_silence"]["start"] == 20.0
    t = r["transcript"]
    assert [(x["kind"], x["start"]) for x in t] == [("speech", 1.0), ("pause", 4.0), ("speech", 12.0), ("pause", 20.0)]
    assert [p["text"] for p in t[0]["parts"] if p["filler"]] == ["Um", "uh"]
    assert "".join(p["text"] for p in t[0]["parts"]) == SEG0.strip()
    assert t[1]["duration_s"] == 8.0 and not t[1]["at_end"] and t[3]["at_end"]


def test_report_without_audio_or_posture_marks_them_not_measured(tmp_path):
    r = load_via_json(tmp_path, make_timeline(audio=False, pose=False))
    o = r["overall"]
    assert o["facing_camera_pct"] == 60.0
    assert o["slouching_pct"] is None and o["leaning_pct"] is None
    assert o["wpm"] is None and o["fillers_per_min"] is None and o["long_pause_count"] is None
    assert [x["wpm"] for x in r["windows"]] == [None, None, None]
    assert [i["metric"] for i in r["feedback"]["improvements"]] == ["facing_camera_pct"]
    assert r["feedback"]["note"] == "This answer measured 1 of the 6 metrics, so the list has fewer than 3 items."


# ---------------------------------------------------------------- feedback

def test_three_improvements_ranked_by_distance_from_the_guideline(tmp_path):
    fb = load_via_json(tmp_path, make_timeline())["feedback"]
    # fillers 7.2/min vs at most 2: 2.6, slouching 25% vs at most 10: 1.5, long pauses 2.4/min vs at most 1: 1.4,
    # wpm 43.2 vs 120 to 160: 0.64, leaning 12.5% vs at most 10: 0.25, facing 60% vs at least 70: 0.143
    assert [i["metric"] for i in fb["ranked"]] == ["fillers_per_min", "slouching_pct", "long_pauses_per_min", "wpm",
                                                  "leaning_pct", "facing_camera_pct"]
    assert [i["distance"] for i in fb["ranked"]] == pytest.approx([2.6, 1.5, 1.4, 0.64, 0.25, 10 / 70], abs=1e-4)
    top = fb["improvements"]
    assert len(top) == 3 and top == fb["ranked"][:3]
    for item in top:
        assert item["guideline"]["kind"] == GUIDELINE_KIND == "coaching guideline, not a measurement"
        assert item["value"] is not None and item["action"] and item["outside_guideline"]
    assert top[0]["value"] == pytest.approx(7.2) and "um or uh" in top[0]["action"]
    assert top[0]["guideline"]["text"] == "at most 2 per minute"
    assert top[1]["guideline"]["text"] == "at most 10% of the answer"


def test_feedback_still_gives_three_when_everything_is_inside_the_guidelines():
    overall = {"facing_camera_pct": 90.0, "slouching_pct": 0.0, "leaning_pct": 0.0, "wpm": 150.0,
               "fillers_per_min": 1.0, "long_pauses_per_min": 0.0}
    fb = build_feedback(overall)
    top = fb["improvements"]
    assert [i["metric"] for i in top] == ["wpm", "facing_camera_pct", "fillers_per_min"]
    assert not any(i["outside_guideline"] for i in top)
    assert top[0]["distance"] == pytest.approx((150 - 160) / 160)
    assert top[0]["action"].startswith("Slow down")  # nearer the upper limit
    slow = build_feedback({**overall, "wpm": 90.0})["improvements"][0]
    assert slow["metric"] == "wpm" and slow["distance"] == pytest.approx(0.25) and "points" in slow["action"]
    assert fb["note"] is None


# ---------------------------------------------------------------- history

def save(tmp_path, session_id, **kw):
    timeline = make_timeline(session_id=session_id, **kw)
    history.save_report(build_report(timeline), tmp_path)
    return timeline


def test_history_lists_sessions_oldest_first_with_the_trend_metrics(tmp_path):
    save(tmp_path, "2026-09-25_120500_t-1", duration=50.0)
    save(tmp_path, "2026-09-25_120000_t-1")
    rows = history.list_sessions(tmp_path)
    assert [r["session_id"] for r in rows] == ["2026-09-25_120000_t-1", "2026-09-25_120500_t-1"]
    assert [m["metric"] for m in history.TREND_METRICS] == ["facing_camera_pct", "wpm", "fillers_per_min",
                                                             "long_pause_count"]
    assert rows[0]["wpm"] == pytest.approx(43.2) and rows[1]["wpm"] == pytest.approx(21.6)  # 18 words in 25 s, 50 s
    assert rows[0]["fillers_per_min"] == pytest.approx(7.2) and rows[1]["long_pause_count"] == 1
    assert history.list_sessions(tmp_path / "missing") == []


def test_history_builds_a_missing_report_from_the_timeline(tmp_path):
    d = tmp_path / "2026-09-25_110000_t-1"
    d.mkdir()
    (d / "timeline.json").write_text(json.dumps(make_timeline(session_id=d.name)), encoding="utf-8")
    assert history.load_report(d.name, tmp_path)["overall"]["facing_camera_pct"] == 60.0
    assert (d / "report.json").exists()
    with pytest.raises(KeyError):
        history.load_report("..\\secrets", tmp_path)
    with pytest.raises(KeyError):
        history.load_report("2026-01-01_000000_none", tmp_path)


def test_server_routes_serve_reports_and_history(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from app import server

    save(tmp_path / "sessions", "2026-09-25_120000_t-1")
    monkeypatch.setitem(server.SETTINGS, "data_dir", tmp_path)
    listing = json.loads(server.api_sessions().body)
    assert len(listing["sessions"]) == 1 and len(listing["trend_metrics"]) == 4
    assert all(g["kind"] == GUIDELINE_KIND for g in listing["guidelines"])
    report = json.loads(server.api_report("2026-09-25_120000_t-1").body)
    assert report["overall"]["facing_camera_pct"] == 60.0
    with pytest.raises(HTTPException) as err:
        server.api_report("nope")
    assert err.value.status_code == 404
