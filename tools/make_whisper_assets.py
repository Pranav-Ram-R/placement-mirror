"""Write the Whisper assets the app needs without transformers: app/audio/assets/.

Runs in the eval venv (transformers). The app reads these files only, so transformers
and tokenizers stay out of the app requirements.

- mel_filters.npy     (80, 201) float64 mel filter bank of WhisperFeatureExtractor
- vocab_bytes.json    token id -> hex of the token's bytes, null for special tokens
- special_tokens.json special token ids and the generation config suppress lists
- prompt_ids.json     decoder prefix (start of transcript, English, transcribe, no
                      timestamps) and the optional filler prompt as prompt ids
- README.md           source checkpoint and how each file was made

Usage: python tools/make_whisper_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "app" / "audio" / "assets"
CHECKPOINT = "openai/whisper-tiny"
# Filler prompt from Task B (eval/fillers/run_filler_test.py PROMPT).
FILLER_PROMPT = "Umm, let me think, like, hmm. Okay, uh, here is what I think."


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte to unicode table used by Whisper's byte level BPE vocabulary."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def main() -> int:
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import GenerationConfig, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(CHECKPOINT)
    gen = GenerationConfig.from_pretrained(CHECKPOINT)
    tok = proc.tokenizer
    fe = proc.feature_extractor
    revision = Path(snapshot_download(CHECKPOINT, local_files_only=True)).name
    OUT.mkdir(parents=True, exist_ok=True)

    mel = np.ascontiguousarray(fe.mel_filters.T)
    assert mel.shape == (80, 201), mel.shape
    np.save(OUT / "mel_filters.npy", mel)

    first_special = tok.eos_token_id  # 50257 <|endoftext|>, every id from here on is special
    decoder = {v: k for k, v in bytes_to_unicode().items()}
    vocab: list[str | None] = []
    for i in range(len(tok)):
        piece = tok.convert_ids_to_tokens(i)
        if i >= first_special:
            vocab.append(None)
            continue
        vocab.append(bytes(decoder[ch] for ch in piece).hex())
    (OUT / "vocab_bytes.json").write_text(json.dumps(vocab) + "\n", encoding="utf-8")

    ids = lambda t: tok.convert_tokens_to_ids(t)  # noqa: E731
    special = {
        "vocab_size": len(tok), "first_special_id": first_special, "eot": tok.eos_token_id,
        "sot": ids("<|startoftranscript|>"), "startofprev": ids("<|startofprev|>"), "en": ids("<|en|>"),
        "transcribe": ids("<|transcribe|>"), "translate": ids("<|translate|>"),
        "notimestamps": ids("<|notimestamps|>"), "timestamp_begin": ids("<|0.00|>"),
        "suppress_tokens": list(gen.suppress_tokens), "begin_suppress_tokens": list(gen.begin_suppress_tokens),
        "max_length": gen.max_length,
    }
    (OUT / "special_tokens.json").write_text(json.dumps(special, indent=1) + "\n", encoding="utf-8")

    prefix = [special["sot"], special["en"], special["transcribe"], special["notimestamps"]]
    prompts = {
        "decoder_prefix": prefix,
        "decoder_prefix_tokens": tok.convert_ids_to_tokens(prefix),
        "filler_prompt": {"text": FILLER_PROMPT, "ids": proc.get_prompt_ids(FILLER_PROMPT).tolist()},
    }
    (OUT / "prompt_ids.json").write_text(json.dumps(prompts, indent=1) + "\n", encoding="utf-8")

    (OUT / "README.md").write_text(f"""# Whisper assets

Written by `tools/make_whisper_assets.py` in the eval venv. The app reads these files so
it does not need transformers or tokenizers at runtime.

Source checkpoint: `{CHECKPOINT}`, Hugging Face revision `{revision}`, the checkpoint the
AI Hub whisper_tiny models are exported from (qai_hub_models whisper_tiny
`WHISPER_VERSION`). Read with transformers {transformers.__version__}. Whisper is MIT licensed.

| file | content |
|---|---|
| `mel_filters.npy` | `WhisperFeatureExtractor.mel_filters` transposed to (80, 201), float64. n_fft {fe.n_fft}, hop {fe.hop_length}, {fe.sampling_rate} Hz, {fe.chunk_length} s window |
| `vocab_bytes.json` | list indexed by token id: the token's bytes as hex (GPT-2 byte to unicode table reversed), `null` for special tokens (id {first_special} and up) |
| `special_tokens.json` | special token ids, `suppress_tokens` and `begin_suppress_tokens` from the generation config |
| `prompt_ids.json` | `decoder_prefix` {prompts['decoder_prefix_tokens']}, and the optional filler prompt from Task B as `WhisperProcessor.get_prompt_ids` ids |
""", encoding="utf-8")
    print(f"wrote {sorted(p.name for p in OUT.iterdir())} from {CHECKPOINT} revision {revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
