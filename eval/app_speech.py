"""The app's own speech path, run offline on a WAV for the evals (pace and fillers).

The audio goes through the same code the app runs live: Silero VAD segmentation
(app.audio.vad.Segmenter with config.AUDIO), then Whisper tiny (app.audio.whisper.Whisper)
on each segment. Models load through ModelRunner, so they run where the runner places
them: on the x86 development machine that is the onnx float32 models on CPU. The shipped
NPU path (precompiled models, float16 math on the HTP) is not what runs here.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from app.audio.vad import Block, Segmenter, SileroVad
from app.audio.whisper import Whisper
from app.config import AUDIO


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (AUDIO.sample_rate, 1, 2):
            raise SystemExit(f"{path}: expected 16 kHz mono 16 bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def segments(runner, audio: np.ndarray) -> list:
    """Speech segments as the app forms them (VAD)."""
    out = []
    seg = Segmenter(SileroVad(runner, AUDIO.sample_rate), out.append, lambda ev: None, AUDIO)
    n = AUDIO.block_samples
    block_s = n / AUDIO.sample_rate
    for i in range(audio.size // n):
        t = (i + 1) * block_s  # stream time at the end of the block, as the capture callback stamps it
        seg.feed(Block(audio[i * n:(i + 1) * n].copy(), t, t))
    seg.flush()
    return out


def transcribe(runner, audio: np.ndarray, use_prompt: bool) -> dict:
    """Each VAD segment transcribed as the app does. wall_s covers VAD and Whisper."""
    t0 = time.perf_counter()
    segs = segments(runner, audio)
    whisper = Whisper(runner, use_prompt)
    results = [whisper.transcribe(s.audio) for s in segs]
    texts = [r.text for r in results]
    return {"texts": texts, "text": " ".join(t for t in texts if t), "segments": len(segs),
            "decoded_text_tokens": sum(len(r.tokens) for r in results), "wall_s": time.perf_counter() - t0}
