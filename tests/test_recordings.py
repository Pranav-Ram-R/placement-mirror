import asyncio
import json
import time
import wave

import numpy as np
import pytest

from app.audio.vad import Block
from app.storage import recordings
from test_session import QUESTION, FakeAudio, calibrate, make_controller

UNKNOWN = b"\x01\xff\xff\xff\xff\xff\xff\xff"  # 8 byte size with every value bit set: unknown size


def el(eid: int, body: bytes = b"", size: bytes | None = None) -> bytes:
    """One EBML element: id, size (1 byte for short bodies unless given), body."""
    head = eid.to_bytes((eid.bit_length() + 7) // 8, "big")
    return head + (size if size is not None else bytes([0x80 | len(body)])) + body


def simple_block(track: int, rel_ms: int) -> bytes:
    return el(0xA3, bytes([0x80 | track]) + rel_ms.to_bytes(2, "big", signed=True) + b"\x80" + b"frame")


def fake_webm() -> bytes:
    """MediaRecorder layout: Segment and Clusters of unknown size. Video frames at 0, 33, 67, 100, 133 ms."""
    tracks = el(0x1654AE6B, el(0xAE, el(0xD7, b"\x01") + el(0x83, b"\x01")))
    info = el(0x1549A966, el(0x2AD7B1, (1_000_000).to_bytes(3, "big")))
    c1 = el(0x1F43B675, size=UNKNOWN) + el(0xE7, b"\x00") + b"".join(simple_block(1, t) for t in (0, 33, 67))
    c2 = el(0x1F43B675, size=UNKNOWN) + el(0xE7, b"\x64") + b"".join(simple_block(1, t) for t in (0, 33))
    return el(0x1A45DFA3, el(0x4282, b"webm")) + el(0x18538067, size=UNKNOWN) + info + tracks + c1 + c2


def test_webm_frame_times_handle_unknown_size_segments_and_clusters(tmp_path):
    assert recordings.webm_frame_times(fake_webm()) == pytest.approx([0.0, 0.033, 0.067, 0.1, 0.133])
    path = tmp_path / "v.webm"
    path.write_bytes(fake_webm())
    t = recordings.webm_timing(path)
    assert t["frames"] == 5 and t["median_frame_interval_s"] == pytest.approx(0.0335, abs=1e-3)
    assert t["duration_s"] == pytest.approx(0.133 + 0.0335, abs=1e-3)
    assert recordings.webm_frame_times(fake_webm()[:-7]) == pytest.approx([0.0, 0.033, 0.067, 0.1])  # truncated


def test_wav_recorder_writes_every_block_and_the_first_sample_time(tmp_path):
    started = []
    rec = recordings.WavRecorder(tmp_path / "a.wav", on_start=started.append)
    now = time.monotonic()
    before = time.time()
    for i in range(3):
        rec.write(Block(np.full(512, 0.5, np.float32), now, 0.0))
    meta = rec.close()
    assert started == [meta["start_epoch_s"]]  # once, with the first block
    assert meta["samples"] == 1536 and meta["duration_s"] == pytest.approx(0.096)
    assert before - 0.2 < meta["start_epoch_s"] < before  # one 32 ms block before the stamp
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 1536)
        assert np.frombuffer(w.readframes(3), "<i2").tolist() == [16383] * 3
    rec.write(Block(np.zeros(512, np.float32), now, 0.0))  # after close: ignored
    assert rec.samples == 1536


def test_save_video_adds_timing_and_the_duration_difference(tmp_path):
    folder = recordings.start_folder(tmp_path, "2026-09-25_120000_t-1", QUESTION)
    meta = recordings.save_video(folder, fake_webm(), start_epoch_ms=1_000_050.0, stop_epoch_ms=1_000_250.0,
                                 mime="video/webm;codecs=vp8")
    assert (folder / "video.webm").read_bytes() == fake_webm()
    assert meta["question_id"] == "t-1" and meta["video"]["frames"] == 5 and "duration_difference_ms" not in meta
    meta = recordings.save_audio(folder, {"duration_s": 0.2, "start_epoch_s": 1000.0})  # the audio lands second
    assert meta["duration_difference_ms"] == pytest.approx((0.1665 - 0.2) * 1000, abs=1.5)
    assert meta["video_start_minus_audio_start_ms"] == pytest.approx(50.0)
    assert [r["id"] for r in recordings.list_recordings(tmp_path)] == ["2026-09-25_120000_t-1"]


class RecordingFakeAudio(FakeAudio):
    def __init__(self, start_mono, recorder):
        super().__init__(start_mono)
        self.recorder = recorder

    def stop(self):
        super().stop()
        if self.recorder is not None:
            for _ in range(10):
                self.recorder.write(Block(np.zeros(512, np.float32), time.monotonic(), 0.0))


def recording_controller(tmp_path, eval_dir):
    c, events, audios = make_controller(tmp_path)
    c.eval_recordings = eval_dir
    c.audio_factory = lambda emit, priority, recorder=None: audios.append(RecordingFakeAudio(10.0, recorder)) or audios[-1]
    return c, events, audios


def test_eval_recording_mode_writes_the_wav_and_metadata_for_the_answer(tmp_path):
    c, events, audios = recording_controller(tmp_path, tmp_path / "recordings")
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    c.command({"type": "start_answer"}, now=10.0)
    answering = next(e for e in events if e.get("state") == "answering")
    sid = answering["session_id"]
    assert answering["recording"] is True and audios[0].recorder is not None
    c.command({"type": "stop_answer"}, now=12.0)
    c.wait_processed(10)
    rec_events = [e for e in events if e["type"] == "eval_recording"]
    assert [e["state"] for e in rec_events] == ["started", "audio_saved"] and rec_events[0]["session_id"] == sid
    folder = tmp_path / "recordings" / sid
    meta = json.loads((folder / "recording.json").read_text(encoding="utf-8"))
    assert meta["session_id"] == sid and meta["question_id"] == "t-1"
    assert meta["audio"]["samples"] == 5120 and meta["audio"]["start_epoch_s"] is not None
    assert (folder / "audio.wav").exists() and c.last_timeline.parent.name == sid  # same id for timeline and recording


def test_normal_mode_passes_no_recorder_and_writes_no_media(tmp_path):
    c, events, audios = recording_controller(tmp_path, None)
    c.command({"type": "select_question", "id": "t-1"})
    calibrate(c)
    c.command({"type": "start_answer"}, now=10.0)
    assert audios[0].recorder is None
    assert next(e for e in events if e.get("state") == "answering")["recording"] is False
    c.command({"type": "stop_answer"}, now=12.0)
    c.wait_processed(10)
    assert not (tmp_path / "recordings").exists()
    assert sorted(p.name for p in c.last_timeline.parent.iterdir()) == ["report.json", "timeline.json"]


def test_server_refuses_video_uploads_unless_record_eval_is_on(monkeypatch):
    from fastapi import HTTPException

    from app import server

    monkeypatch.setitem(server.SETTINGS, "record_eval", False)
    assert json.loads(server.api_config().body)["record_eval"] is False
    with pytest.raises(HTTPException) as err:
        asyncio.run(server.api_eval_video("2026-09-25_120000_t-1", request=None))
    assert err.value.status_code == 404
