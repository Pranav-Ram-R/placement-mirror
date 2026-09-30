"""Render the submission files to images for review, through Microsoft Word and PowerPoint.

Word and PowerPoint (COM automation from PowerShell) export each file to PDF, PyMuPDF turns
every page into a PNG and Pillow puts the pages of each file on one contact sheet. Output
goes to --out (a scratch folder, not committed). Windows with Office only. Run in .venv-eval.

Usage: python tools/render_docs.py --out <folder>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCX = REPO / "docs" / "description" / "PlacementMirror_Description.docx"
PPTX = REPO / "docs" / "deck" / "PlacementMirror.pptx"

PS = r"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$d = $w.Documents.Open('{docx}', $false, $true)
$d.ExportAsFixedFormat('{docx_pdf}', 17)
$d.Close($false)
$w.Quit()
$p = New-Object -ComObject PowerPoint.Application
$pr = $p.Presentations.Open('{pptx}', $true, $false, $false)
$pr.SaveAs('{pptx_pdf}', 32)
$pr.Close()
$p.Quit()
"""


def pages(pdf: Path, out: Path, stem: str, zoom: float) -> list[Path]:
    import pymupdf

    paths = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            path = out / f"{stem}_{i:02d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(path)
            paths.append(path)
    return paths


def sheet(images: list[Path], cols: int, path: Path, width: int) -> None:
    from PIL import Image, ImageDraw

    ims = [Image.open(p).convert("RGB") for p in images]
    tw = width // cols
    th = int(tw * ims[0].height / ims[0].width)
    rows = -(-len(ims) // cols)
    pad = 16
    out = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + pad + 22) + pad), "white")
    draw = ImageDraw.Draw(out)
    for i, im in enumerate(ims):
        x, y = pad + (i % cols) * (tw + pad), pad + (i // cols) * (th + pad + 22)
        out.paste(im.resize((tw, th)), (x, y + 22))
        draw.rectangle([x - 1, y + 21, x + tw, y + 22 + th], outline=(160, 160, 160))
        draw.text((x, y + 4), images[i].stem, fill=(40, 40, 40))
    out.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    docx_pdf, pptx_pdf = out / "description.pdf", out / "deck.pdf"
    for p in (docx_pdf, pptx_pdf):
        p.unlink(missing_ok=True)
    script = PS.format(docx=DOCX, docx_pdf=docx_pdf, pptx=PPTX, pptx_pdf=pptx_pdf)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    doc_pages = pages(docx_pdf, out, "description_page", 2.0)
    deck_pages = pages(pptx_pdf, out, "deck_slide", 1.5)
    sheet(doc_pages, 2, out / "description_contact_sheet.png", 1800)
    sheet(deck_pages, 2, out / "deck_contact_sheet.png", 2000)
    print(f"Description: {len(doc_pages)} pages. Deck: {len(deck_pages)} slides. Images in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
