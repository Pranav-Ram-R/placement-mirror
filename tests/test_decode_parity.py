"""Our decode loop on the AI Hub onnx whisper_tiny (CPU) against transformers generate.

Runs where transformers and torch are installed (dev or eval venv) and the models from
aihub.fetch_models are in models/. On a mismatch the assertion message shows both token
lists and both texts.
"""

from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from app.audio.whisper import Whisper, assets  # noqa: E402
from app.runtime.runner import MANIFEST, ModelRunner  # noqa: E402


class CpuOnly:
    available = False
    reason = "test: CPU only"


def jfk_audio() -> np.ndarray:
    found = sorted((Path.home() / ".qaihm").glob("**/audio/jfk.npz"))
    if not found:
        pytest.skip("qai_hub_models jfk.npz sample not in the local asset cache")
    return np.load(found[0])["audio"].astype(np.float32)


@pytest.fixture(scope="module")
def ours():
    if not MANIFEST.exists():
        pytest.skip("models/manifest.json missing, run python -m aihub.fetch_models")
    runner = ModelRunner(qnn=CpuOnly())
    return {p: Whisper(runner, use_prompt=p) for p in (False, True)}


@pytest.fixture(scope="module")
def hf():
    proc = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = transformers.WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
    return proc, model


CLIPS = {"jfk 11 s": lambda a: a, "jfk first 5 s": lambda a: a[: 5 * 16000]}


@pytest.mark.parametrize("use_prompt", [False, True])
@pytest.mark.parametrize("clip", list(CLIPS))
def test_decode_matches_transformers(ours, hf, clip, use_prompt):
    audio = CLIPS[clip](jfk_audio())
    proc, model = hf
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    kwargs = {}
    if use_prompt:
        kwargs["prompt_ids"] = torch.tensor(assets()["prompts"]["filler_prompt"]["ids"])
    with torch.no_grad():
        ids = model.generate(feats, language="en", task="transcribe", **kwargs)[0].tolist()
    eot = assets()["special"]["eot"]
    ref_tokens = [t for t in ids if t != eot]
    ref_text = proc.decode(ids, skip_special_tokens=True)
    got = ours[use_prompt].transcribe(audio)
    assert got.stopped == "end of text"
    assert got.tokens == ref_tokens, f"tokens differ\n ours {got.tokens}\n hf   {ref_tokens}"
    assert got.text == ref_text, f"text differs\n ours {got.text!r}\n hf   {ref_text!r}"
