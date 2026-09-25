"""Download models-v1.zip from the GitHub Release, check its sha256 and extract it (Task L).

The sha256 must match packaging/models-v1.sha256 (written by tools/package_models.py).
Extract into the repository root for development, or into the packaged app's _internal
folder. Used by CI. The app itself never downloads anything.

Usage: python tools/fetch_models.py [--dest .] [--url URL] [--zip FILE]
  --zip FILE  use this local zip instead of downloading (it is still checked)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.package_models import HASH_FILE, sha256_file  # noqa: E402

URL = "https://github.com/Pranav-Ram-R/placement-mirror/releases/download/models-v1/models-v1.zip"


def expected_sha256() -> str:
    return HASH_FILE.read_text(encoding="utf-8").split()[0]


def download(url: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(target, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                done += len(chunk)
            print(f"Downloaded {done} bytes{f' of {total}' if total else ''} from {url}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit(f"{url} is not published (HTTP 404). The author uploads it once, "
                             "see tools/package_models.py.") from None
        raise


def extract(zip_path: Path, dest: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "models":
                raise SystemExit(f"Unexpected path in the models zip: {n}")
        zf.extractall(dest)
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--zip", type=Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = args.zip or Path(tmp) / "models-v1.zip"
        if args.zip is None:
            download(args.url, zip_path)
        digest, want = sha256_file(zip_path), expected_sha256()
        if digest != want:
            raise SystemExit(f"sha256 of {zip_path.name} is {digest}, packaging/models-v1.sha256 expects {want}")
        count = extract(zip_path, args.dest)
    print(f"sha256 matches ({digest}). Extracted {count} files to {args.dest / 'models'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
