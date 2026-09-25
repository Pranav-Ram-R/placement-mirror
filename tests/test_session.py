import json
import threading
from collections import Counter

import pytest

from app.config import SessionConfig
from app.session.controller import TIMELINE_SCHEMA, SessionController
from app.session.machine import InvalidTransition, SessionMachine, State
from app.session.questions import BANK, find_question, load_bank
from test_pipeline import FakeRunner, frame

QUESTION = {"id": "t-1", "category": "hr", "text": "Tell me about yourself.", "type": "behavioral",
            "suggested_time_s": 30}


# ---------------------------------------------------------------- question bank

def test_question_bank_has_30_questions_10_per_category_and_is_marked_draft():
    bank = load_bank(BANK)
    assert bank["status"] == "DRAFT, needs author review"
    assert len(bank["questions"]) == 30
    assert Counter(q["category"] for q in bank["questions"]) == {"hr": 10, "sde": 10, "hardware": 10}
    assert {q["type"] for q in bank["questions"] if q["category"] == "hr"} == {"behavioral"}
    assert {q["type"] for q in bank["questions"] if q["category"] != "hr"} == {"technical"}
    for q in bank["questions"]:
        assert set(q) == {"id", "category", "text", "type", "suggested_time_s"}
        assert "—" not in q["text"] and ";" not in q["text"]  # CLAUDE.md rule 8


def test_question_bank_rejects_bad_entries(tmp_path):
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    bank["questions"][0]["type"] = "trivia"
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bank(path)
    with pytest.raises(KeyError):
        find_question(load_bank(BANK), "nope")


# ---------------------------------------------------------------- state machine

def test_state_machine_runs_the_session_flow_and_rejects_other_steps():
    m = SessionMachine()
    with pytest.raises(InvalidTransition):
        m.fire("start")
    assert m.fire("select", QUESTION) == State.CALIBRATE
    with pytest.raises(InvalidTransition):
        m.fire("start")  # not calibrated
    assert m.fire("calibrated") == State.READY
    assert m.fire("start") == State.ANSWERING
    with pytest.raises(InvalidTransition):
        m.fire("select", QUESTION)  # no question change during an answer
    assert m.fire("stop") == State.PROCESSING
    assert m.fire("processed") == State.REPORT
    assert m.fire("recalibrate") == State.CALIBRATE
    assert [h[1] for h in m.history] == ["choose_question", "calibrate", "ready", "answering", "processing", "report",
                                         "calibrate"]


# ---------------------------------------------------------------- controller

class FakeSource:
    def describe(self):
        return "fake audio"


class FakeAudio:
    """Stands in for AudioPipeline: records and pauses in the same shape."""

    def __init__(self, start_mono):
        self.source = FakeSource()
        self.t = start_mono
        self.records, self.pauses = [], []
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        t = self.t
        self.records = [
            {"index": 0, "start_mono": t + 0.5, "end_mono": t + 2.0, "text": " Um, I am a student.",
             "fillers": ["Um"], "word_count": 5, "closed_by": "silence"},
            {"index": 1, "start_mono": t + 5.0, "end_mono": t + 7.5, "text": " I like hardware.",
             "fillers": [], "word_count": 3, "closed_by": "silence"},
        ]
        self.pauses = [{"state": "started", "since_mono": t + 2.0}, {"state": "ended", "duration_s": 3.0},
                       {"state": "started", "since_mono": t + 7.5}]


def make_controller(tmp_path, cfg=SessionConfig(), audio=True):
    events, audios = [], []

    def factory(emit, priority):
        a = FakeAudio(start_mono=10.0)
        audios.append(a)
        return a

    bank = {"status": "test", "categories": {"hr": "HR"}, "questions": [QUESTION]}
    c = SessionController(FakeRunner(), events.append, bank=bank, sessions_dir=tmp_path / "sessions",
                          audio_factory=factory if audio else None, cfg=cfg)
    return c, events, audios


def states(events):
    return [e["state"] for e in events if e["type"] == "session_state"]


def calibrate(c, t0=1.0):
    c.command({"type": "calibrate"}, now=t0)
    for i in range(20):
        c.vision.process(frame(i, t0 + i * 0.1))
    c.vision.process(frame(20, t0 + 3.1))  # after the 3 s window


def test_full_session_saves_a_timeline_covering_the_whole_answer(tmp_path):
    c, events, audios = make_controller(tmp_path)
    c.command({"type": "calibrate"}, now=0.5)
    assert events[-1]["type"] == "error"  # no question chosen yet
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    assert c.machine.state == State.READY
    c.command({"type": "start_answer"}, now=10.0)
    assert audios[0].started and events[-1]["limit_s"] == 90  # 30 s suggested + 60 s
    for i in range(100):  # 10 s of frames during the answer
        c.vision.process(frame(100 + i, 10.0 + i * 0.1))
    c.command({"type": "stop_answer"}, now=19.95)
    c.wait_processed(10)
    assert states(events)[-3:] == ["answering", "processing", "report"]
    report_event = events[-1]
    tl = json.loads(open(report_event["timeline"], encoding="utf-8").read())
    assert tl["schema"] == TIMELINE_SCHEMA and tl["question"]["id"] == "t-1"
    assert tl["answer"]["stopped_by"] == "user" and tl["answer"]["duration_s"] == pytest.approx(9.95)
    frames = tl["frames"]
    assert len(frames) == 100 and frames[0]["t"] == 0.0 and frames[-1]["t"] == pytest.approx(9.9)
    assert set(frames[0]) >= {"t", "facing", "yaw", "pitch", "slouching", "leaning"}
    assert all(f["facing"] is True for f in frames)  # the fake face looks at the camera
    assert [(s["start"], s["end"], s["word_count"]) for s in tl["segments"]] == [(0.5, 2.0, 5), (5.0, 7.5, 3)]
    assert tl["segments"][0]["fillers"] == ["Um"] and tl["segments"][0]["text"] == " Um, I am a student."
    assert tl["pauses"][0] == {"start": 2.0, "end": 5.0, "duration_s": 3.0}
    assert tl["pauses"][1]["start"] == 7.5 and tl["pauses"][1]["open_at_stop"] is True
    assert tl["summary"]["frames"] == 100 and tl["summary"]["segments"] == 2 and tl["summary"]["words"] == 8
    assert report_event["summary"] == tl["summary"]


def test_answer_stops_by_itself_at_the_limit(tmp_path):
    c, events, _ = make_controller(tmp_path, cfg=SessionConfig(auto_stop_extra_s=0.3 - 30))
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    c.command({"type": "start_answer"}, now=10.0)
    done = threading.Event()
    for _ in range(100):
        if c.machine.state == State.REPORT:
            done.set()
            break
        threading.Event().wait(0.05)
    assert done.is_set()
    tl = json.loads(open(events[-1]["timeline"], encoding="utf-8").read())
    assert tl["answer"]["stopped_by"].startswith("auto stop at 0.3 s")


def test_calibration_without_a_face_keeps_the_calibrate_step(tmp_path):
    c, events, _ = make_controller(tmp_path)
    c.command({"type": "select_question", "id": "t-1"})
    c.command({"type": "calibrate"}, now=1.0)
    for i in range(20):
        c.vision.process(frame(i, 1.0 + i * 0.1, face=False))  # no detector input, no face
    c.vision.process(frame(20, 4.1, face=False))
    assert c.machine.state == State.CALIBRATE
    assert "did not see a face" in events[-1]["message"]


def test_end_of_replay_video_is_recorded_as_the_stop_reason(tmp_path):
    c, events, _ = make_controller(tmp_path)
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    c.command({"type": "start_answer"}, now=10.0)
    c.command({"type": "stop_answer", "reason": "end of replay video"}, now=12.0)
    c.wait_processed(10)
    assert json.loads(c.last_timeline.read_text(encoding="utf-8"))["answer"]["stopped_by"] == "end of replay video"


def test_close_during_an_answer_saves_it(tmp_path):
    c, events, audios = make_controller(tmp_path)
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    c.command({"type": "start_answer"}, now=10.0)
    c.close()
    assert c.machine.state == State.REPORT and audios[0].stopped
    assert c.last_timeline.exists()
    assert json.loads(c.last_timeline.read_text(encoding="utf-8"))["answer"]["stopped_by"] == "connection closed"
