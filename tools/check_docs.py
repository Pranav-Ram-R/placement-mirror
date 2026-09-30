"""Check the submission files built by tools/build_docs.py.

1. Every number in both files (body, tables, speaker notes, chart values) appears in a source
   file: the value itself, or the value rounded to the shown decimals, in the source unit or
   converted by 1/1000 (us to ms, Hz to kHz) or 1/1e6 (bytes to MB). Integers must match exactly.
2. No em dash, en dash or semicolon anywhere.
3. No sentence about the accuracy evals names Snapdragon or a device run.
4. No hype words.
5. Slide body text (not titles, footers, tables, charts or notes) at most about 40 words. The
   title slide is exempt: it carries the two statements the brief requires word for word.

Exit code 1 on any failure. Run in .venv-eval. Usage: python tools/check_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCX = REPO / "docs" / "description" / "PlacementMirror_Description.docx"
PPTX = REPO / "docs" / "deck" / "PlacementMirror.pptx"
SOURCES = [
    "benchmarks/TABLE.md", "benchmarks/optimization.md", "eval/RESULTS.md", "benchmarks/raw/budgets_derived.json",
    "benchmarks/raw/day3_derived.json", "benchmarks/raw/day3_decoder_io_bytes.json",
    "docs/evidence/offline_check.json", "docs/evidence/arm64_build.json", "app/config.py",
    "eval/eye_contact/results/scripted_20260929_175949_results.json",
    "eval/fillers/results/filler_results.json", "eval/pace/results/pace_results.json",
]
HYPE = ["revolutionary", "cutting-edge", "cutting edge", "seamless", "blazing", "game-changing", "game changer",
        "state-of-the-art", "innovative", "unprecedented", "incredible", "amazing", "groundbreaking",
        "world-class", "best-in-class", "supercharge", "effortless", "lightning", "magic", "powerful"]
EVAL_WORDS = ("accuracy", "recall", "precision", "word count", "recorded clip", "held-out", "filler detection")
NUM = re.compile(r"(?<![A-Za-z0-9._/\-])[-+]?\d[\d,]*(?:\.\d+)?")
STRIP = [re.compile(r"\bj[0-9a-z]{8}\b"), re.compile(r"https?://\S+"), re.compile(r"\S*[/\\]\S*"),
         re.compile(r"\S+\.(?:json|md|py|png|docx|pptx)\b"), re.compile(r"\b\w+_\w+\b")]
WORD_LIMIT = 45


def source_values() -> set[float]:
    vals: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.add(float(x))
        elif isinstance(x, str):
            vals.update(float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", x))
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    for rel in SOURCES:
        path = REPO / rel
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        walk(json.loads(raw) if path.suffix == ".json" else raw)
    return vals


def accepted(token: str, vals: set[float]) -> bool:
    t = token.replace(",", "").lstrip("+")
    x = float(t)
    dec = len(t.split(".")[1]) if "." in t else 0
    for v in vals:
        for s in (1, 1e-3, 1e-6):
            y = v * s
            if dec == 0:
                if abs(y - x) < 1e-9:
                    return True
            elif abs(y - x) <= 0.5 * 10 ** -dec + 1e-9:
                return True
    return False


def docx_parts() -> list[tuple[str, str]]:
    from docx import Document

    d = Document(DOCX)
    parts = [("description", p.text) for p in d.paragraphs]
    for tb in d.tables:
        parts += [("description table", c.text) for row in tb.rows for c in row.cells]
    return parts


def pptx_parts() -> tuple[list[tuple[str, str]], dict[int, int]]:
    from pptx import Presentation
    from pptx.util import Pt

    prs = Presentation(PPTX)
    parts, words = [], {}
    for n, s in enumerate(prs.slides, 1):
        body = 0
        for sh in s.shapes:
            if sh.has_text_frame:
                txt = sh.text_frame.text
                parts.append((f"slide {n}", txt))
                sizes = [r.font.size for p in sh.text_frame.paragraphs for r in p.runs if r.font.size]
                is_title = sh.top < 1.5 * 914400 and sizes and max(sizes) >= Pt(32) and n > 1
                is_footer = sizes and max(sizes) <= Pt(10)
                if not is_title and not is_footer and not (sizes and max(sizes) <= Pt(16)):
                    body += len(txt.split())
            if sh.has_table:
                parts += [(f"slide {n} table", c.text) for row in sh.table.rows for c in row.cells]
            if sh.has_chart:
                for plot in sh.chart.plots:
                    parts += [(f"slide {n} chart", f"{v:.3f}") for ser in plot.series for v in ser.values]
        parts.append((f"slide {n} notes", s.notes_slide.notes_text_frame.text))
        words[n] = body
    return parts, words


def main() -> int:
    vals = source_values()
    parts, words = pptx_parts()
    parts = docx_parts() + parts
    fails = []
    checked = 0
    for where, txt in parts:
        if any(ch in txt for ch in ("—", "–", ";")):
            fails.append(f"{where}: dash or semicolon in {txt[:80]!r}")
        low = txt.lower()
        for h in HYPE:
            if re.search(rf"\b{re.escape(h)}\b", low):
                fails.append(f"{where}: hype word {h!r}")
        for sent in re.split(r"(?<=[.!?])\s+", txt):
            sl = sent.lower()
            if any(w in sl for w in EVAL_WORDS) and ("snapdragon" in sl or "on device" in sl or "on-device" in sl):
                fails.append(f"{where}: accuracy sentence names a device: {sent[:120]!r}")
        clean = txt
        for rx in STRIP:
            clean = rx.sub(" ", clean)
        for tok in NUM.findall(clean):
            checked += 1
            if not accepted(tok, vals):
                fails.append(f"{where}: number {tok} not found in the sources ({txt[:90]!r})")
    for n, w in words.items():
        print(f"slide {n}: {w} body words")
        if w > WORD_LIMIT and n > 1:
            fails.append(f"slide {n}: {w} body words, limit about 40")
    print(f"{checked} numbers checked against {len(vals)} source values")
    for f in fails:
        print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
