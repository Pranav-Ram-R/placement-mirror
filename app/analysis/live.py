"""Live speaking indicators from transcript segments.

WPM = words in the last 30 s / 30 * 60 (config.wpm_window_s). A segment's words are
spread evenly over the segment's time span. Fillers are counted with a case-insensitive
whole word match for um, umm, uh, uhh, hmm and mm.
"""

from __future__ import annotations

import re
import threading

import numpy as np

from app.config import AUDIO, AudioConfig

FILLER_RE = re.compile(r"\b(?:um|umm|uh|uhh|hmm|mm)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def find_fillers(text: str) -> list[str]:
    return [m.group(0) for m in FILLER_RE.finditer(text)]


class LiveAnalysis:
    def __init__(self, cfg: AudioConfig = AUDIO):
        self.cfg = cfg
        self.word_times: list[float] = []
        self.fillers: list[dict] = []
        self.words_total = 0
        self._lock = threading.Lock()

    def add_segment(self, text: str, start: float, end: float) -> None:
        n = count_words(text)
        with self._lock:
            if n:
                self.word_times.extend(np.linspace(start, end, n + 2)[1:-1].tolist())
            self.words_total += n
            self.fillers.extend({"word": w, "at": end} for w in find_fillers(text))

    def wpm(self, now: float) -> float:
        with self._lock:
            recent = sum(1 for t in self.word_times if now - self.cfg.wpm_window_s < t <= now)
        return recent / self.cfg.wpm_window_s * 60.0

    def indicators(self, now: float) -> dict:
        wpm = self.wpm(now)
        with self._lock:
            return {"type": "speech", "wpm": wpm, "words_total": self.words_total,
                    "filler_count": len(self.fillers),
                    "fillers_by_word": {w: sum(1 for f in self.fillers if f["word"].lower() == w)
                                        for w in sorted({f["word"].lower() for f in self.fillers})}}
