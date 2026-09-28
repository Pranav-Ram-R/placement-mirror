"""Pace eval (Day 3 item 2c): the app's word count against a manual count.

For each row of eval/pace/manual_counts.csv (columns clip, words), the clip's WAV goes
through the app's own speech path offline (eval/app_speech.py: Silero VAD segments, then
Whisper tiny with the prompt as in config.AUDIO.use_prompt) and
app.analysis.live.count_words. Models load through ModelRunner, so on the x86 development
machine they run on CPU.

clip: an eval recording id (eval/recordings/<id>/audio.wav) or a path to a 16 kHz mono
16 bit WAV.
Error % = (app words - manual words) / manual words * 100 (Derived).

Results go to eval/pace/results/pace_results.json. Transcripts are the speaker's own words
and go to eval/pace/results/pace_transcripts_local.json, which is gitignored.

Usage: python eval/pace/run_pace_test.py [--counts eval/pace/manual_counts.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.analysis.live import count_words  # noqa: E402
from app.audio.pipeline import AUDIO_MODELS  # noqa: E402
from app.config import AUDIO  # noqa: E402
from app.runtime.runner import ModelRunner  # noqa: E402
from app.storage.recordings import RECORDINGS_DIR  # noqa: E402
from benchmarks.schema import Source  # noqa: E402
from eval.app_speech import read_wav, transcribe  # noqa: E402

HERE = Path(__file__).resolve().parent


def clip_path(clip: str) -> Path:
    rec = RECORDINGS_DIR / clip / "audio.wav"
    return rec if rec.exists() else Path(clip)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=Path, default=HERE / "manual_counts.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.counts, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {args.counts}")
    runner = ModelRunner()
    units = {name: runner.load(name).compute_unit for name in AUDIO_MODELS}
    results, transcripts = [], []
    for row in rows:
        clip, manual = row["clip"].strip(), int(row["words"])
        path = clip_path(clip)
        audio = read_wav(path)
        texts = transcribe(runner, audio, AUDIO.use_prompt)["texts"]
        app_words = sum(count_words(t) for t in texts)
        results.append({
            "clip": clip, "audio_s": round(audio.size / AUDIO.sample_rate, 3), "segments": len(texts),
            "app_words": app_words, "manual_words": manual,
            "error_pct": round((app_words - manual) / manual * 100, 2) if manual else None,
            "error_formula": "Derived: (app_words - manual_words) / manual_words * 100",
        })
        transcripts.append({"clip": clip, "segments": texts})
        print(f"{clip}: app {app_words} words, manual {manual}, error {results[-1]['error_pct']} % "
              f"({len(texts)} segments, {results[-1]['audio_s']} s)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": Source.LOCAL_X86_CPU.value, "machine": platform.machine(), "processor": platform.processor(),
        "compute_units": units, "use_prompt": AUDIO.use_prompt, "counts_file": str(args.counts),
        "word_counter": "app.analysis.live.count_words (regex [A-Za-z0-9]+(?:'[A-Za-z]+)?)",
        "vad": {k: getattr(AUDIO, k) for k in ("segment_end_silence_ms", "segment_max_s", "speech_pad_ms",
                                               "vad_threshold", "vad_neg_threshold")},
        "results": results,
    }
    (args.out_dir / "pace_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "pace_transcripts_local.json").write_text(json.dumps(transcripts, indent=1) + "\n",
                                                              encoding="utf-8")
    print(f"Wrote {args.out_dir / 'pace_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
