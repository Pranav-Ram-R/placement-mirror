"""Audio pipeline for one session: capture, VAD, segment queue, ASR worker.

- capture: MicCapture (sounddevice callback thread) or FileSource puts 32 ms blocks on a
  queue
- VAD thread: Silero VAD per block, segmentation, pause events
- ASR worker: Whisper on each closed segment, transcript events, then live indicators
  (WPM, fillers) through app.analysis.live. Indicators are also sent every
  config.indicator_interval_s so WPM falls during silence.

Per segment timings (time.perf_counter()):
- vad_latency: segment closed by the VAD minus the end of its last speech block. It
  includes the config.segment_end_silence_ms wait.
- mel, encoder, decoder per call (one token per call), token count
- end_to_end_delay: transcript ready minus the end of the last speech block
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from app.analysis.live import LiveAnalysis, find_fillers
from app.audio.capture import MicCapture
from app.audio.vad import Segment, Segmenter, SileroVad
from app.audio.whisper import Whisper
from app.config import AUDIO, AudioConfig

AUDIO_MODELS = ("silero_vad", "whisper_tiny_encoder", "whisper_tiny_decoder")


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


class AudioPipeline:
    def __init__(self, runner, emit: Callable[[dict], None], cfg: AudioConfig = AUDIO, source_factory=None):
        self.runner = runner
        self.emit = emit
        self.cfg = cfg
        self.blocks: queue.Queue = queue.Queue()
        self.segments: queue.Queue = queue.Queue()
        self.segmenter = Segmenter(SileroVad(runner, cfg.sample_rate), self.segments.put, self._event, cfg)
        self.whisper = Whisper(runner, cfg.use_prompt)
        self.live = LiveAnalysis(cfg)
        self.source = source_factory(self.blocks.put) if source_factory else MicCapture(self.blocks.put, cfg)
        self.t0 = time.monotonic()
        self.records: list[dict] = []
        self.pauses: list[dict] = []
        self.blocks_seen = 0
        self._running = False
        self._vad_stop = threading.Event()
        self._vad_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []

    def _rel(self, t_mono: float) -> float:
        return t_mono - self.t0

    def _event(self, event: dict) -> None:
        if event.get("type") == "pause":
            event = {**event, "at_s": self._rel(event.pop("at"))}
            if "since" in event:
                event["since_s"] = self._rel(event.pop("since"))
            self.pauses.append(event)
        self.emit(event)

    def start(self) -> None:
        self._running = True
        self.t0 = time.monotonic()
        self._vad_thread = threading.Thread(target=self._vad_loop, name="audio-vad", daemon=True)
        asr = threading.Thread(target=self._asr_loop, name="audio-asr", daemon=True)
        self._vad_thread.start()
        asr.start()
        self._threads = [asr]
        self.source.start()

    def stop(self, drain_s: float = 30.0) -> None:
        self.source.stop()
        deadline = time.monotonic() + drain_s
        while not self.blocks.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        self._vad_stop.set()
        self._vad_thread.join(timeout=drain_s)
        self.segmenter.flush()
        self._running = False
        for t in self._threads:
            t.join(timeout=drain_s)

    def _vad_loop(self) -> None:
        while not self._vad_stop.is_set():
            try:
                block = self.blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            self.blocks_seen += 1
            self.segmenter.feed(block)

    def _asr_loop(self) -> None:
        last_indicators = 0.0
        while self._running or not self.segments.empty():
            try:
                seg = self.segments.get(timeout=self.cfg.indicator_interval_s)
            except queue.Empty:
                seg = None
            if seg is not None:
                self._transcribe(seg)
            now = time.monotonic()
            if now - last_indicators >= self.cfg.indicator_interval_s or seg is not None:
                last_indicators = now
                self.emit({**self.live.indicators(now), "at_s": self._rel(now)})

    def _transcribe(self, seg: Segment) -> None:
        t_start = time.perf_counter()
        tr = self.whisper.transcribe(seg.audio)
        t_done = time.perf_counter()
        self.live.add_segment(tr.text, seg.start_mono, seg.speech_end_mono)
        calls = len(tr.decoder_call_s)
        record = {
            "index": seg.index, "start_s": self._rel(seg.start_mono), "end_s": self._rel(seg.speech_end_mono),
            "audio_s": seg.duration_s, "closed_by": seg.reason, "text": tr.text, "fillers": find_fillers(tr.text),
            "token_count": len(tr.tokens), "decoder_calls": calls, "stopped": tr.stopped,
            "vad_latency_ms": (seg.closed_perf - seg.speech_end_perf) * 1000,
            "asr_queue_wait_ms": (t_start - seg.closed_perf) * 1000,
            "mel_ms": tr.mel_s * 1000, "encoder_ms": tr.encoder_s * 1000,
            "decoder_ms_total": tr.decoder_s * 1000,
            "decoder_ms_per_token": tr.decoder_s * 1000 / calls if calls else float("nan"),
            "end_to_end_delay_ms": (t_done - seg.speech_end_perf) * 1000,
        }
        self.records.append(record)
        self.emit({"type": "transcript", **{k: v for k, v in record.items() if k != "fillers"},
                   "fillers": record["fillers"]})

    def report(self) -> dict:
        keys = ("vad_latency_ms", "encoder_ms", "decoder_ms_per_token", "token_count", "end_to_end_delay_ms",
                "mel_ms", "asr_queue_wait_ms")
        return {
            "source": self.source.describe(), "duration_s": time.monotonic() - self.t0,
            "blocks": self.blocks_seen, "segments": self.records, "pauses": self.pauses,
            "vad_call_ms": percentiles([s * 1000 for s in self.segmenter.stats.call_s]),
            "summary": {k: percentiles([r[k] for r in self.records]) for k in keys},
            "words_total": self.live.words_total, "filler_count": len(self.live.fillers),
            "use_prompt": self.cfg.use_prompt,
        }
