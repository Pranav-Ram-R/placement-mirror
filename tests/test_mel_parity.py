"""app.audio.mel against transformers WhisperFeatureExtractor. Runs in the eval venv."""

from pathlib import Path

import numpy as np
import pytest

from app.audio.mel import N_FRAMES, log_mel, pad_or_trim

transformers = pytest.importorskip("transformers")

SR = 16000


def jfk_audio() -> np.ndarray | None:
    """qai_hub_models Whisper sample (11 s of speech) when it is in the local asset cache."""
    found = sorted((Path.home() / ".qaihm").glob("**/audio/jfk.npz"))
    return np.load(found[0])["audio"].astype(np.float32) if found else None


def signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(8 * SR) / SR
    out = {
        "chirp and noise 8 s": (0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t) + rng.normal(0, 0.02, t.size)).astype(np.float32),
        "silence 2 s": np.zeros(2 * SR, np.float32),
        "noise 35 s (trimmed)": rng.normal(0, 0.1, 35 * SR).astype(np.float32),
    }
    jfk = jfk_audio()
    if jfk is not None:
        out["jfk speech 11 s"] = jfk
    return out


@pytest.fixture(scope="module")
def extractor():
    return transformers.WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")


@pytest.mark.parametrize("name", list(signals()))
def test_log_mel_matches_whisper_feature_extractor(extractor, name):
    audio = signals()[name]
    ours = log_mel(audio)
    assert ours.shape == (80, N_FRAMES) and ours.dtype == np.float32
    default = extractor(audio, sampling_rate=SR, return_tensors="np")["input_features"][0]
    assert np.abs(ours - default).max() < 1e-4
    numpy_path = extractor._np_extract_fbank_features(pad_or_trim(audio)[None], "cpu")[0]
    assert np.abs(ours - numpy_path).max() < 1e-4
