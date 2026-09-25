"""Streaming Silero VAD (v6.2.3 ONNX through ModelRunner) and speech segmentation.

Model call, as silero-vad's OnnxWrapper does it at 16 kHz: input is the last 64 samples
of the previous call followed by the new 512 samples, state is (2, 1, 128) carried from
call to call, sr is 16000 as int64.

Segmentation per 32 ms block:
- speech starts when the probability reaches config.vad_threshold and continues while it
  stays at or above config.vad_neg_threshold
- a segment ends after config.segment_end_silence_ms of silence, or when it reaches
  config.segment_max_s (then a new segment starts at once if speech continues)
- config.speech_pad_ms of audio is kept before the start and after the end of speech
- a pause event is sent once a silence after speech passes config.pause_min_s, and a
  pause end event with its length when speech starts again

Times are the capture times of the blocks: time.monotonic() for the session clock and
time.perf_counter() for durations (time.monotonic() on Windows has 15.625 ms resolution).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.config import AUDIO, AudioConfig

MODEL = "silero_vad"
CONTEXT = 64


@dataclass
class Block:
    samples: np.ndarray  # (512,) float32
    t_mono: float  # time.monotonic() when the block was captured (its end)
    t_perf: float  # time.perf_counter() at the same moment


@dataclass
class Segment:
    index: int
    audio: np.ndarray
    start_mono: float  # start of the first kept block
    speech_end_mono: float  # end of the last block with speech
    speech_end_perf: float
    closed_perf: float  # when the VAD closed the segment
    reason: str  # "silence", "max length" or "end of input"
    blocks: int = 0

    @property
    def duration_s(self) -> float:
        return self.audio.size / AUDIO.sample_rate


@dataclass
class VadStats:
    call_s: list[float] = field(default_factory=list)


class SileroVad:
    """Silero VAD with streaming state."""

    def __init__(self, runner, sample_rate: int = 16000):
        self.runner = runner
        self.sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)

    def __call__(self, block: np.ndarray) -> tuple[float, float]:
        x = np.concatenate([self.context, block.astype(np.float32, copy=False)])[None]
        out, seconds = self.runner.run(MODEL, {"input": x, "state": self.state, "sr": self.sr})
        self.state = out["stateN"]
        self.context = x[0, -CONTEXT:]
        return float(out["output"].reshape(-1)[0]), seconds


class Segmenter:
    def __init__(self, vad, emit_segment, emit_event, cfg: AudioConfig = AUDIO):
        self.vad = vad
        self.emit_segment = emit_segment
        self.emit_event = emit_event
        self.cfg = cfg
        self.block_s = cfg.block_samples / cfg.sample_rate
        self.pad_blocks = max(1, math.ceil(cfg.speech_pad_ms / 1000 / self.block_s))
        self.end_blocks = math.ceil(cfg.segment_end_silence_ms / 1000 / self.block_s)
        self.max_blocks = int(cfg.segment_max_s / self.block_s)
        self.pre_roll: deque[Block] = deque(maxlen=self.pad_blocks)
        self.stats = VadStats()
        self.segments = 0
        self.in_speech = False
        self.current: list[Block] = []
        self.silent_run = 0
        self.last_speech: Block | None = None
        self.pause_started = False

    def feed(self, block: Block) -> float:
        prob, seconds = self.vad(block.samples)
        self.stats.call_s.append(seconds)
        if not self.in_speech:
            if prob >= self.cfg.vad_threshold:
                self._start(block)
            else:
                self.pre_roll.append(block)
                self._check_pause(block)
            return prob
        self.current.append(block)
        if prob >= self.cfg.vad_neg_threshold:
            self.silent_run = 0
            self.last_speech = block
        else:
            self.silent_run += 1
        if self.silent_run >= self.end_blocks:
            self._close(block, "silence")
            self.in_speech = False
        elif len(self.current) >= self.max_blocks:
            self._close(block, "max length")
            self._start_continuation()
        return prob

    def flush(self) -> None:
        """Close a segment still open when the input stops."""
        if self.in_speech and self.current:
            self._close(self.current[-1], "end of input")
            self.in_speech = False

    def _start(self, block: Block) -> None:
        if self.pause_started and self.last_speech is not None:
            self.emit_event({"type": "pause", "state": "ended", "at": block.t_mono,
                             "duration_s": block.t_mono - self.block_s - self.last_speech.t_mono})
        self.pause_started = False
        self.in_speech = True
        self.current = list(self.pre_roll) + [block]
        self.pre_roll.clear()
        self.silent_run = 0
        self.last_speech = block

    def _start_continuation(self) -> None:
        self.current = []
        self.silent_run = 0

    def _check_pause(self, block: Block) -> None:
        if self.last_speech is None or self.pause_started:
            return
        silence = block.t_mono - self.last_speech.t_mono
        if silence >= self.cfg.pause_min_s:
            self.pause_started = True
            self.emit_event({"type": "pause", "state": "started", "at": block.t_mono, "since": self.last_speech.t_mono})

    def _close(self, block: Block, reason: str) -> None:
        blocks = self.current
        if reason == "silence":
            # keep speech_pad_ms after the last speech block, drop the rest of the silence
            last = next((i for i, b in enumerate(blocks) if b is self.last_speech), None)
            if last is None:
                return  # continuation after a max length cut that held no more speech
            blocks = blocks[: last + 1 + self.pad_blocks]
        audio = np.concatenate([b.samples for b in blocks])
        first = blocks[0]
        end = self.last_speech if reason == "silence" else blocks[-1]
        seg = Segment(index=self.segments, audio=audio, start_mono=first.t_mono - self.block_s,
                      speech_end_mono=end.t_mono, speech_end_perf=end.t_perf, closed_perf=time.perf_counter(),
                      reason=reason, blocks=len(blocks))
        self.segments += 1
        self.emit_segment(seg)
