"""Whisper filler capture kill test (Task B).

Runs the float reference models on the local x86 CPU with Hugging Face transformers,
using the checkpoints behind the whisper_tiny and distil_whisper models we profiled
on AI Hub. The checkpoint ids are read from the qai_hub_models source.

Two conditions per model: no prompt, and one fixed disfluent prompt. For each clip
it records the transcript, the fillers found, the decoded token counts and the wall
time. Filler recall per model and condition = total found / total labeled.

Timings are saved as Measurement records (Source.LOCAL_X86_CPU, runtime
"transformers", precision "float32"). Transcripts, token counts and recall go to
<out-dir>/filler_results.json.

Decoding setup, identical for every model and condition. It mirrors how the app will
run Whisper, once per speech segment with the same prompt each time:
- a clip longer than 30 s is split into windows of at most 30 s, each split at the
  quietest 100 ms frame between 20 s and 30 s into the current window
- each window is decoded on its own (short-form), with the prompt when the condition
  has one, no timestamps and no conditioning on the previous window's text
- greedy (num_beams=1, do_sample=False), no temperature fallback
- the multilingual checkpoint is forced to language="en", task="transcribe"
Transformers' built-in long-form mode was not used: it only applies a prompt to every
window with condition_on_prev_tokens=True, and with that setting distil-small.en
dropped the last 20 s of a 46 s test clip.

Clips come from --clips-dir (every .wav, clip id = file name without .wav) and, with
--source recordings or both, from the app's eval recordings (server flag --record-eval):
the audio.wav of every folder in --recordings-dir, clip id = the recording's session id.
labels.csv has one row per clip id (clip column) with um_count, uh_count and hmm_count.

Usage:
  python eval/fillers/run_filler_test.py [--clips-dir eval/fillers] [--labels eval/fillers/labels.csv]
                                         [--source clips|recordings|both]
                                         [--recordings-dir eval/recordings]
                                         [--out-dir eval/fillers/results]
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import importlib.util
import json
import platform
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.storage.recordings import RECORDINGS_DIR, list_recordings  # noqa: E402
from benchmarks.schema import Measurement, Source, save_json  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT = "Umm, let me think, like, hmm. Okay, uh, here is what I think."
CONDITIONS = {"no_prompt": None, "disfluent_prompt": PROMPT}
FILLER_RE = re.compile(r"\b(um|umm|uh|uhh|hmm|mm)\b", re.IGNORECASE)
FORM_TO_TYPE = {"um": "um", "umm": "um", "uh": "uh", "uhh": "uh", "hmm": "hmm", "mm": "hmm"}
TYPES = ("um", "uh", "hmm")
RATE = 16000

# qai_hub_models model id -> (module holding the checkpoint constant, constant name)
MODELS = {
    "whisper_tiny": ("qai_hub_models.models.whisper_tiny.model", "WHISPER_VERSION"),
    "distil_whisper": ("qai_hub_models.models.distil_whisper.model", "DISTIL_WHISPER_VERSION"),
}


def checkpoint_from_source(module: str, constant: str) -> tuple[str, str]:
    """Read a string constant from the qai_hub_models source file without importing it."""
    path = importlib.util.find_spec(module).origin
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == constant for t in node.targets):
            return ast.literal_eval(node.value), path
    raise SystemExit(f"{constant} not found in {path}")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if fmt != (RATE, 1, 2):
            raise SystemExit(f"{path}: expected 16 kHz mono 16-bit PCM, got rate/channels/bytes {fmt}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def find_clips(source: str, clips_dir: Path, recordings_dir: Path) -> list[tuple[str, str, Path]]:
    """(clip id, name shown in results, WAV path) for every clip of the chosen source."""
    clips = []
    if source in ("clips", "both"):
        clips += [(p.stem, p.name, p) for p in sorted(clips_dir.glob("*.wav"))]
    if source in ("recordings", "both"):
        clips += [(r["id"], f"recordings/{r['id']}/audio.wav", r["audio"]) for r in list_recordings(recordings_dir)
                  if r["audio"] is not None]
    return clips


def read_labels(path: Path) -> dict[str, dict[str, int]]:
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[Path(row["clip"].strip()).stem] = {t: int(row[f"{t}_count"]) for t in TYPES}
    return labels


def count_fillers(text: str) -> tuple[list[str], dict[str, int]]:
    found = [m.group(0) for m in FILLER_RE.finditer(text)]
    by_type = {t: 0 for t in TYPES}
    for f in found:
        by_type[FORM_TO_TYPE[f.lower()]] += 1
    return found, by_type


def token_counts(ids: list[int], tokenizer) -> dict[str, int]:
    """Split the returned ids into prompt, forced start tokens and decoded tokens."""
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    first_timestamp = tokenizer.convert_tokens_to_ids("<|0.00|>")
    special = set(tokenizer.all_special_ids)
    prompt = forced = 0
    decoded = ids
    if sot in ids:
        # Everything before SOT is the prompt. SOT and the special tokens right after it
        # (language, task, notimestamps) are forced, not decoded.
        i = ids.index(sot)
        j = i + 1
        while j < len(ids) and ids[j] in special and ids[j] != eot and ids[j] < first_timestamp:
            j += 1
        prompt, forced, decoded = i, j - i, ids[j:]
    return {
        "prompt_tokens_in_output": prompt,
        "forced_start_tokens": forced,
        "decoded_tokens": len(decoded),
        "decoded_text_tokens": sum(1 for t in decoded if t < eot),
    }


def split_windows(audio: np.ndarray, max_s: float = 30.0, min_s: float = 20.0, frame_s: float = 0.1) -> list[tuple[int, int]]:
    """Split into windows of at most max_s seconds at the quietest frame after min_s seconds."""
    windows, start, n = [], 0, len(audio)
    frame = int(frame_s * RATE)
    while n - start > max_s * RATE:
        lo, hi = start + int(min_s * RATE), start + int(max_s * RATE)
        frames = audio[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        quietest = int(np.argmin(np.sqrt((frames ** 2).mean(axis=1))))
        split = lo + quietest * frame + frame // 2
        windows.append((start, split))
        start = split
    windows.append((start, n))
    return windows


def transcribe(model, processor, audio: np.ndarray, prompt: str | None, english_only: bool) -> dict:
    kwargs = dict(num_beams=1, do_sample=False, return_timestamps=False, return_dict_in_generate=True)
    if not english_only:
        kwargs.update(language="en", task="transcribe")
    if prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    texts, ids_per_window = [], []
    windows = split_windows(audio)
    for a, b in windows:
        inputs = processor(audio[a:b], sampling_rate=RATE, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            out = model.generate(inputs.input_features, attention_mask=inputs.attention_mask, **kwargs)
        ids = out["sequences"][0].tolist()  # full sequence: prompt, forced start tokens, decoded tokens
        ids_per_window.append(ids)
        texts.append(processor.decode(ids, skip_special_tokens=True).strip())
    wall = time.perf_counter() - t0
    return {"text": " ".join(t for t in texts if t), "ids_per_window": ids_per_window, "wall_s": wall,
            "windows_s": [[round(a / RATE, 2), round(b / RATE, 2)] for a, b in windows]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=str(HERE))
    ap.add_argument("--labels", default=str(HERE / "labels.csv"))
    ap.add_argument("--source", choices=["clips", "recordings", "both"], default="clips")
    ap.add_argument("--recordings-dir", default=str(RECORDINGS_DIR))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import transformers

    clips = find_clips(args.source, Path(args.clips_dir), Path(args.recordings_dir))
    labels = read_labels(Path(args.labels))
    missing = [cid for cid, _, _ in clips if cid not in labels]
    if not clips:
        raise SystemExit(f"No clips for --source {args.source} in {args.clips_dir} or {args.recordings_dir}")
    if missing:
        raise SystemExit(f"No labels for {missing} in {args.labels}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    print(json.dumps(env, indent=1))
    runs, measurements, models_info = [], [], {}
    for model_id, (module, constant) in MODELS.items():
        checkpoint, source_file = checkpoint_from_source(module, constant)
        processor = WhisperProcessor.from_pretrained(checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        english_only = checkpoint.endswith(".en")
        models_info[model_id] = {"checkpoint": checkpoint, "read_from": f"{constant} in {source_file}",
                                 "parameter_dtypes": dtypes, "english_only": english_only}
        print(f"\n== {model_id}: {checkpoint} (from {constant}), parameters {dtypes}")
        # One untimed warm-up call per model so lazy initialization is not in the first timing.
        transcribe(model, processor, read_wav(clips[0][2])[: 5 * RATE], None, english_only)
        for condition, prompt in CONDITIONS.items():
            for cid, name, path in clips:
                audio = read_wav(path)
                r = transcribe(model, processor, audio, prompt, english_only)
                found, by_type = count_fillers(r["text"])
                lab = labels[cid]
                per_window = [token_counts(ids, processor.tokenizer) for ids in r["ids_per_window"]]
                tokens = {k: sum(w[k] for w in per_window) for k in per_window[0]}
                runs.append({
                    "model": model_id, "checkpoint": checkpoint, "condition": condition, "clip": name,
                    "audio_seconds": round(len(audio) / RATE, 3), "windows_s": r["windows_s"],
                    "tokens_per_window": per_window, "transcript": r["text"], "fillers_found": found,
                    "found_by_type": by_type, "found_total": len(found), "labeled_by_type": lab,
                    "labeled_total": sum(lab.values()), **tokens,
                })
                measurements.append(Measurement(
                    model=checkpoint, metric="transcribe_wall_time", value=r["wall_s"] * 1000, unit="ms",
                    source=Source.LOCAL_X86_CPU, runtime="transformers", compute_unit="CPU", precision="float32",
                    notes=(f"clip {name} ({len(audio) / RATE:.1f} s audio), condition {condition}, "
                           f"{len(r['windows_s'])} window(s), feature extraction + generate + decode, greedy, torch {torch.__version__}, "
                           f"transformers {transformers.__version__}, {torch.get_num_threads()} torch threads, "
                           f"CPU {platform.processor()}, one untimed warm-up call per model before timing"),
                ))
                print(f"   {condition:17} {name:22} labeled {sum(lab.values()):3} found {len(found):3} "
                      f"decoded tokens {tokens['decoded_tokens']:4}  {r['text'][:90]}")

    summary = []
    for model_id in MODELS:
        for condition in CONDITIONS:
            rs = [r for r in runs if r["model"] == model_id and r["condition"] == condition]
            found = sum(r["found_total"] for r in rs)
            labeled = sum(r["labeled_total"] for r in rs)
            summary.append({
                "model": model_id, "checkpoint": models_info[model_id]["checkpoint"], "condition": condition,
                "clips": len(rs), "total_found": found, "total_labeled": labeled,
                "recall": (found / labeled) if labeled else None,
                "recall_formula": "Derived: total_found / total_labeled (found can exceed labeled)",
                "found_by_type": {t: sum(r["found_by_type"][t] for r in rs) for t in TYPES},
                "labeled_by_type": {t: sum(r["labeled_by_type"][t] for r in rs) for t in TYPES},
            })

    results = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": Source.LOCAL_X86_CPU.value, "runtime": "transformers", "precision": "float32",
        "environment": env, "models": models_info, "prompt": PROMPT, "conditions": list(CONDITIONS),
        "filler_regex": FILLER_RE.pattern + " (case-insensitive)", "form_to_type": FORM_TO_TYPE,
        "decoding": {"num_beams": 1, "do_sample": False, "return_timestamps": False,
                     "windows": "at most 30 s, split at the quietest 100 ms frame between 20 s and 30 s",
                     "prompt": "prompt_ids on every window for the disfluent_prompt condition",
                     "previous_window_text": "not used",
                     "multilingual_forced": {"language": "en", "task": "transcribe"}},
        "clips_dir": str(Path(args.clips_dir)), "source": args.source, "recordings_dir": str(Path(args.recordings_dir)),
        "labels_file": str(Path(args.labels)),
        "timings_file": str(out_dir / "filler_timings.json"),
        "summary": summary, "runs": runs,
    }
    (out_dir / "filler_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    save_json(measurements, out_dir / "filler_timings.json")

    print("\nSummary (Source: local x86 CPU, transformers, float32)")
    print(f"| model | condition | clips | labeled | found | recall (Derived: found / labeled) |")
    print("|---|---|---|---|---|---|")
    for s in summary:
        recall = f"{s['recall']:.3f}" if s["recall"] is not None else "n/a"
        print(f"| {s['model']} | {s['condition']} | {s['clips']} | {s['total_labeled']} | {s['total_found']} | {recall} |")
    print("\nDecoded tokens per clip (excluding prompt and forced start tokens)")
    conds = [(m, c) for m in MODELS for c in CONDITIONS]
    print("| clip | " + " | ".join(f"{m} {c}" for m, c in conds) + " |")
    print("|---|" + "---|" * len(conds))
    for _, name, _ in clips:
        row = [next(r["decoded_tokens"] for r in runs if r["clip"] == name and r["model"] == m and r["condition"] == c)
               for m, c in conds]
        print(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    print(f"\nWrote {out_dir / 'filler_results.json'} and {len(measurements)} timing measurements to "
          f"{out_dir / 'filler_timings.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
