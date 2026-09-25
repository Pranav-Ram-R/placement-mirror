"""Whisper tiny transcription with the AI Hub encoder and decoder through ModelRunner.

Matches the AI Hub model signatures and the qai_hub_models hf_whisper app loop:
- encoder: input_features (1, 80, 3000) -> k/v_cache_cross_0..3
- decoder: one token per call. input_ids (1, 1) int32, position_ids (1,) int32,
  attention_mask (1, 1, 1, 200) with -100 for masked positions, self attention caches
  k/v_cache_self_0..3_in of 199 positions fed back from the _out outputs, and the cross
  caches. At call n the mask position 199 - n is opened. The cache holds 199 tokens, so
  decoding stops after 199 calls or at end of text.

Decoding is greedy. The decoder prefix is start of transcript, English, transcribe, no
timestamps (app/audio/assets/prompt_ids.json), fed one token per call, as transformers
forces it for language="en", task="transcribe". The generation config's suppress_tokens
are masked at every step and begin_suppress_tokens at the first generated step, as
transformers does. With use_prompt (config, default off) the Task B filler prompt ids
(start of previous, prompt text) go before the prefix. Text comes from the token bytes
in vocab_bytes.json. Special tokens are dropped.

Input and cache dtypes follow the loaded session (float16 for the precompiled models,
float32 for the onnx models).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.audio.mel import log_mel

ASSETS = Path(__file__).resolve().parent / "assets"
ENCODER, DECODER = "whisper_tiny_encoder", "whisper_tiny_decoder"
MASK_NEG = -100.0  # qai_hub_models hf_whisper MASK_NEG


@lru_cache(maxsize=1)
def assets() -> dict:
    load = lambda name: json.loads((ASSETS / name).read_text(encoding="utf-8"))  # noqa: E731
    vocab = [bytes.fromhex(v) if v is not None else None for v in load("vocab_bytes.json")]
    return {"vocab": vocab, "special": load("special_tokens.json"), "prompts": load("prompt_ids.json")}


def detokenize(tokens: list[int]) -> str:
    a = assets()
    first_special = a["special"]["first_special_id"]
    return b"".join(a["vocab"][t] for t in tokens if t < first_special).decode("utf-8", errors="replace")


def numpy_dtype(onnx_type: str):
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32,
            "tensor(int64)": np.int64}[onnx_type]


@dataclass
class Transcript:
    text: str
    tokens: list[int]  # generated tokens, without the prefix and end of text
    prefix_len: int
    mel_s: float
    encoder_s: float
    decoder_call_s: list[float] = field(default_factory=list)
    stopped: str = ""  # "end of text" or "cache full"

    @property
    def decoder_s(self) -> float:
        return float(sum(self.decoder_call_s))


class Whisper:
    def __init__(self, runner, use_prompt: bool = False):
        self.runner = runner
        self.use_prompt = use_prompt
        a = assets()
        self.special = a["special"]
        self.prefix = list(a["prompts"]["decoder_prefix"])
        if use_prompt:
            self.prefix = list(a["prompts"]["filler_prompt"]["ids"]) + self.prefix
        self.enc_specs = {n: (s, numpy_dtype(t)) for n, s, t in runner.input_specs(ENCODER)}
        self.dec_specs = {n: (s, numpy_dtype(t)) for n, s, t in runner.input_specs(DECODER)}
        self.mask_len = int(self.dec_specs["attention_mask"][0][-1])
        self.n_calls_max = self.mask_len - 1
        if len(self.prefix) >= self.n_calls_max:
            raise ValueError(f"prefix of {len(self.prefix)} tokens does not fit the {self.n_calls_max} position cache")
        vocab = self.special["vocab_size"]
        self.suppress = np.zeros(vocab, bool)
        self.suppress[self.special["suppress_tokens"]] = True
        self.begin_suppress = self.suppress.copy()
        self.begin_suppress[self.special["begin_suppress_tokens"]] = True

    def encode(self, mel: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        shape, dtype = self.enc_specs["input_features"]
        out, seconds = self.runner.run(ENCODER, {"input_features": mel.reshape(shape).astype(dtype)})
        return out, seconds

    def decode(self, cross: dict[str, np.ndarray]) -> tuple[list[int], list[float], str]:
        feeds: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in self.dec_specs.items():
            if name.startswith(("k_cache_self", "v_cache_self")):
                feeds[name] = np.zeros(shape, dtype)
            elif name.startswith(("k_cache_cross", "v_cache_cross")):
                feeds[name] = cross[name].astype(dtype, copy=False)
        mask_dtype = self.dec_specs["attention_mask"][1]
        mask = np.full((1, 1, 1, self.mask_len), MASK_NEG, mask_dtype)
        ids_dtype = self.dec_specs["input_ids"][1]
        pos_dtype = self.dec_specs["position_ids"][1]
        sequence = list(self.prefix)
        call_s: list[float] = []
        eot = self.special["eot"]
        stopped = "cache full"
        for n in range(self.n_calls_max):
            mask[0, 0, 0, self.mask_len - n - 1] = 0.0
            feeds["input_ids"] = np.array([[sequence[n]]], ids_dtype)
            feeds["position_ids"] = np.array([n], pos_dtype)
            feeds["attention_mask"] = mask
            out, seconds = self.runner.run(DECODER, feeds)
            call_s.append(seconds)
            for name in list(feeds):
                if name.startswith(("k_cache_self", "v_cache_self")):
                    feeds[name] = out[name[:-3] + "_out"]
            if n + 1 < len(sequence):
                continue  # next token is given by the prefix
            logits = out["logits"].reshape(-1).astype(np.float32)
            logits[self.begin_suppress if n + 1 == len(self.prefix) else self.suppress] = -np.inf
            token = int(np.argmax(logits))
            if token == eot:
                stopped = "end of text"
                break
            sequence.append(token)
        return sequence[len(self.prefix):], call_s, stopped

    def transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        mel = log_mel(audio)
        mel_s = time.perf_counter() - t0
        cross, encoder_s = self.encode(mel)
        tokens, call_s, stopped = self.decode(cross)
        return Transcript(detokenize(tokens), tokens, len(self.prefix), mel_s, encoder_s, call_s, stopped)
