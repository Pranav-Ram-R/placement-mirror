"""Audio capture: sounddevice at 16 kHz mono in 32 ms (512 sample) blocks.

Each block is stamped with time.monotonic() and time.perf_counter() in the stream
callback, which runs when the block is complete. FileSource plays a recording through the
same interface in real time, for tests and automated runs without a microphone.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from app.audio.vad import Block
from app.config import AUDIO, AudioConfig


class MicCapture:
    def __init__(self, on_block: Callable[[Block], None], cfg: AudioConfig = AUDIO, device=None):
        self.on_block = on_block
        self.cfg = cfg
        self.device = device
        self.stream = None
        self.status_flags: list[str] = []

    def _callback(self, indata, frames, time_info, status) -> None:
        t_mono, t_perf = time.monotonic(), time.perf_counter()
        if status:
            self.status_flags.append(str(status))
        self.on_block(Block(indata[:, 0].copy(), t_mono, t_perf))

    def start(self) -> None:
        import sounddevice as sd

        self.stream = sd.InputStream(samplerate=self.cfg.sample_rate, channels=1, dtype="float32",
                                     blocksize=self.cfg.block_samples, device=self.device, callback=self._callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def describe(self) -> str:
        import sounddevice as sd

        info = sd.query_devices(self.device, "input")
        return f"microphone {info['name']}"


class FileSource:
    """Plays 16 kHz mono float32 audio as 32 ms blocks at real time pace."""

    def __init__(self, on_block: Callable[[Block], None], audio: np.ndarray, label: str, cfg: AudioConfig = AUDIO,
                 speed: float = 1.0):
        self.on_block = on_block
        self.audio = np.asarray(audio, np.float32).reshape(-1)
        self.label = label
        self.cfg = cfg
        self.speed = speed
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.done = threading.Event()

    def _run(self) -> None:
        n = self.cfg.block_samples
        block_s = n / self.cfg.sample_rate / self.speed
        start = time.perf_counter()
        for i in range(self.audio.size // n):
            due = start + (i + 1) * block_s
            while not self._stop.is_set() and time.perf_counter() < due:
                time.sleep(min(0.004, max(0.0, due - time.perf_counter())))
            if self._stop.is_set():
                break
            self.on_block(Block(self.audio[i * n:(i + 1) * n].copy(), time.monotonic(), time.perf_counter()))
        self.done.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="audio-file", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def describe(self) -> str:
        return self.label
