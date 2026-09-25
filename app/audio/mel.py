"""Whisper log-mel spectrogram in numpy.

Same steps as transformers WhisperFeatureExtractor (openai/whisper-tiny settings):
pad with zeros or trim to 30 s at 16 kHz, STFT with n_fft 400, hop 160, a periodic Hann
window and reflect padding of n_fft / 2 on both sides (torch.stft center=True), power
spectrum, drop the last frame, 80 mel filters (app/audio/assets/mel_filters.npy),
log10 with a floor of 1e-10, clamp to 8 below the maximum, then (x + 4) / 4.
Output (80, 3000) float32.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
CHUNK_S = 30
N_SAMPLES = CHUNK_S * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP
ASSETS = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    return np.load(ASSETS / "mel_filters.npy")


@lru_cache(maxsize=1)
def hann_window() -> np.ndarray:
    n = np.arange(N_FFT)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic, as torch.hann_window


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.size))


def log_mel(audio: np.ndarray) -> np.ndarray:
    """(80, 3000) float32 log-mel features of up to 30 s of 16 kHz mono audio."""
    x = pad_or_trim(audio).astype(np.float64)
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = sliding_window_view(x, N_FFT)[::HOP]  # (3001, 400)
    spec = np.fft.rfft(frames * hann_window(), axis=-1)
    power = spec.real ** 2 + spec.imag ** 2
    mel = mel_filters() @ power[:-1].T  # (80, 3000)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
