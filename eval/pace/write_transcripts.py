"""Transcripts for the manual pace word counts.

For each clip in eval/pace/manual_counts.csv, the clip's WAV goes through the app's own
speech path (eval/app_speech.py, prompt as in config.AUDIO.use_prompt) and the text is
written to eval/pace/transcripts/<clip>_local.txt, which is gitignored. The author corrects
each file while listening, so it holds what was said, fillers included.

--count prints the word count of each corrected file with app.analysis.live.count_words,
the rule the app counts with, for the words column of manual_counts.csv.

Usage: python eval/pace/write_transcripts.py [--count]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.analysis.live import count_words  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "transcripts"


def clips(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["clip"].strip() for row in csv.DictReader(f)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=Path, default=HERE / "manual_counts.csv")
    ap.add_argument("--count", action="store_true", help="print the word count of each corrected transcript")
    args = ap.parse_args()
    names = clips(args.counts)
    if args.count:
        for name in names:
            path = OUT / f"{name}_local.txt"
            print(f"{name},{count_words(path.read_text(encoding='utf-8'))}")
        return 0

    from app.audio.pipeline import AUDIO_MODELS
    from app.config import AUDIO
    from app.runtime.runner import ModelRunner
    from eval.app_speech import read_wav, transcribe
    from eval.pace.run_pace_test import clip_path

    runner = ModelRunner()
    for name in AUDIO_MODELS:
        runner.load(name)
    OUT.mkdir(exist_ok=True)
    for name in names:
        r = transcribe(runner, read_wav(clip_path(name)), AUDIO.use_prompt)
        path = OUT / f"{name}_local.txt"
        path.write_text("\n".join(t for t in r["texts"] if t) + "\n", encoding="utf-8")
        print(f"Wrote {path.relative_to(REPO)} ({r['segments']} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
