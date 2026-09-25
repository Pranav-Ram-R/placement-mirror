import numpy as np
import pytest

from app.analysis.live import LiveAnalysis, count_words, find_fillers
from app.audio.vad import Block, Segmenter
from app.audio.whisper import assets, detokenize
from app.config import AUDIO

BLOCK_S = AUDIO.block_samples / AUDIO.sample_rate  # 0.032


class ScriptedVad:
    """Returns one scripted speech probability per block."""

    def __init__(self, probs):
        self.probs = list(probs)

    def __call__(self, samples):
        return self.probs.pop(0), 0.0001


def run(probs):
    segments, events = [], []
    seg = Segmenter(ScriptedVad(probs), segments.append, events.append)
    for i in range(len(probs)):
        t = (i + 1) * BLOCK_S
        seg.feed(Block(np.full(AUDIO.block_samples, i, np.float32), t, t))
    return seg, segments, events


def blocks(seconds):
    return int(round(seconds / BLOCK_S))


PAD = 7  # 200 ms of padding in 32 ms blocks, rounded up
END = 32  # 1000 ms of silence in 32 ms blocks


def test_segment_ends_after_1_s_of_silence_with_200_ms_padding():
    assert (AUDIO.speech_pad_ms, AUDIO.segment_end_silence_ms, AUDIO.segment_max_s) == (200, 1000, 20.0)
    probs = [0.1] * 10 + [0.9] * blocks(1.0) + [0.1] * (END - 1)
    _, segments, _ = run(probs)
    assert segments == []  # 31 silent blocks (992 ms) do not end it
    probs = [0.1] * 10 + [0.9] * blocks(1.0) + [0.1] * END
    _, segments, _ = run(probs)
    assert len(segments) == 1
    s = segments[0]
    assert s.reason == "silence"
    speech_blocks = blocks(1.0)
    # 7 pre-roll blocks, the speech, 7 blocks of padding after it
    assert s.blocks == PAD + speech_blocks + PAD
    assert s.audio[0] == 10 - PAD and s.audio[-1] == 10 + speech_blocks + PAD - 1  # block values are their index
    last_speech = 10 + speech_blocks - 1
    assert s.speech_end_mono == pytest.approx((last_speech + 1) * BLOCK_S)


def test_short_dips_do_not_end_a_segment_and_hysteresis_keeps_speech():
    probs = [0.9] * 20 + [0.1] * 25 + [0.4] * 5 + [0.9] * 20 + [0.1] * blocks(1.1)
    _, segments, _ = run(probs)
    assert len(segments) == 1  # 800 ms dip and 0.4 (above neg_threshold 0.35) keep it open


def test_segment_is_cut_at_20_s_and_continues():
    probs = [0.9] * blocks(30.0) + [0.1] * blocks(1.1)
    _, segments, _ = run(probs)
    assert [s.reason for s in segments] == ["max length", "silence"]
    assert segments[0].audio.size / AUDIO.sample_rate == pytest.approx(20.0, abs=BLOCK_S * 2)


def test_pause_events_for_silence_over_2_s():
    probs = [0.1] * blocks(3.0) + [0.9] * 20 + [0.1] * blocks(2.5) + [0.9] * 20 + [0.1] * blocks(1.1)
    _, segments, events = run(probs)
    assert len(segments) == 2
    started = [e for e in events if e["state"] == "started"]
    ended = [e for e in events if e["state"] == "ended"]
    assert len(started) == 1 and len(ended) == 1  # the silence before any speech is not a pause
    assert ended[0]["duration_s"] == pytest.approx(2.5, abs=2 * BLOCK_S)


def test_flush_closes_open_speech():
    seg, segments, _ = run([0.9] * 30)
    assert segments == []
    seg.flush()
    assert len(segments) == 1 and segments[0].reason == "end of input"


def test_detokenize_uses_token_bytes_and_drops_special_tokens():
    special = assets()["special"]
    assert detokenize([special["sot"], 400, 370, 452, 7177, 6280, 13, special["eot"]]) == " And so my fellow Americans."
    assert len(assets()["vocab"]) == special["vocab_size"]


def test_fillers_and_words():
    text = "Umm, so I think, uh, that hmm we MM should. Mmm and umbrella and summary. Uhh."
    assert find_fillers(text) == ["Umm", "uh", "hmm", "MM", "Uhh"]
    assert count_words("It's a test, isn't it?") == 5


def test_wpm_counts_words_in_the_last_30_s():
    live = LiveAnalysis()
    live.add_segment("one two three four five six seven eight nine ten", 0.0, 10.0)
    live.add_segment("a b c d e", 40.0, 45.0)
    assert live.wpm(10.0) == pytest.approx(10 / 30 * 60)
    assert live.wpm(45.0) == pytest.approx(5 / 30 * 60)  # the first segment is older than 30 s
    ind = live.indicators(45.0)
    assert ind["words_total"] == 15 and ind["filler_count"] == 0
