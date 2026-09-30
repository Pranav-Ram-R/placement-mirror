"""Rerun the filler and pace evals when their labels are filled, then rebuild RESULTS.md and both
submission files, and check them.

1. eval/fillers/labels.csv: when every row has um_count, uh_count and hmm_count, run
   eval/fillers/run_filler_test.py --source recordings (.venv-eval). Otherwise skip it.
2. eval/pace/manual_counts.csv: when every row has words, run eval/pace/run_pace_test.py
   (.venv-eval). Otherwise skip it.
3. eval/write_results.py regenerates eval/RESULTS.md. A section without a results file says
   "not yet evaluated".
4. tools/build_docs.py rebuilds the description and the deck, tools/check_docs.py checks them.
5. With --render DIR, tools/render_docs.py renders both files and the contact sheets into DIR.

Usage: python tools/refresh_docs.py [--render <scratch folder>]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_PY = REPO / ".venv-eval" / "Scripts" / "python.exe"
EYE = REPO / "eval" / "eye_contact" / "results" / "scripted_20260929_175949_results.json"


def filled(path: Path, columns: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return bool(rows) and all((row.get(c) or "").strip() for row in rows for c in columns)


def run(*args: str) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run([str(EVAL_PY), *args], cwd=REPO, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path)
    args = ap.parse_args()
    if filled(REPO / "eval" / "fillers" / "labels.csv", ("um_count", "uh_count", "hmm_count")):
        run("eval/fillers/run_filler_test.py", "--source", "recordings")
    else:
        print("eval/fillers/labels.csv has empty counts, filler test skipped (not yet evaluated)")
    if filled(REPO / "eval" / "pace" / "manual_counts.csv", ("words",)):
        run("eval/pace/run_pace_test.py")
    else:
        print("eval/pace/manual_counts.csv has empty counts, pace test skipped (not yet evaluated)")
    run("eval/write_results.py", "--eye-contact", str(EYE))
    run("tools/build_docs.py", "--eye-contact", str(EYE))
    run("tools/check_docs.py")
    if args.render:
        run("tools/render_docs.py", "--out", str(args.render))
    return 0


if __name__ == "__main__":
    sys.exit(main())
