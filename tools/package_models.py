"""Package the model files as models-v1.zip, for the GitHub Release asset CI downloads (Task L).

Models are not in git. The zip holds models/manifest.json and every file the manifest
lists, each checked against its sha256 in the manifest first. Paths in the zip match the
repository (models/<model>/<runtime>/...), so extracting it into the repository root, or
into the packaged app's _internal folder, puts every file where app.runtime.runner looks.

The zip is deterministic (sorted entries, fixed timestamps, deflate level 9), so the same
model files give the same zip. Its sha256 goes to packaging/models-v1.sha256, which
tools/fetch_models.py checks the downloaded asset against.

The author uploads it once (CI never creates releases):
  gh release create models-v1 dist/models-v1.zip --title "Models v1" --notes "Model files for CI builds"

Usage: python tools/package_models.py [--out dist/models-v1.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "models/manifest.json"
HASH_FILE = ROOT / "packaging" / "models-v1.sha256"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_files(root: Path = ROOT) -> list[dict]:
    """Every file the manifest lists, once, with its sha256 and size."""
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    files = {}
    for entry in manifest["models"]:
        for f in entry["files"]:
            files[f["path"]] = f
    return [files[p] for p in sorted(files)]


def build_zip(out: Path, root: Path = ROOT) -> str:
    files = manifest_files(root)
    bad = [f["path"] for f in files if not (root / f["path"]).exists() or sha256_file(root / f["path"]) != f["sha256"]]
    if bad:
        raise SystemExit(f"Missing or changed model files (sha256 differs from {MANIFEST}): {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        for rel in [MANIFEST] + [f["path"] for f in files]:
            info = zipfile.ZipInfo(rel, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (root / rel).read_bytes(), compresslevel=9)
    return sha256_file(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "models-v1.zip")
    args = ap.parse_args()
    digest = build_zip(args.out)
    HASH_FILE.write_text(f"{digest}  models-v1.zip\n", encoding="utf-8", newline="\n")
    files = manifest_files()
    print(f"Wrote {args.out}: {args.out.stat().st_size} bytes, {len(files) + 1} files, sha256 {digest}")
    print(f"Wrote {HASH_FILE.relative_to(ROOT)}. Commit it, then upload the zip once:")
    print(f'  gh release create models-v1 "{args.out}" --title "Models v1" --notes "Model files for CI builds"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
