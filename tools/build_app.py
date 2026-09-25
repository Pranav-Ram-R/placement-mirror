"""Build the packaged app with PyInstaller: dist/PlacementMirror/ (one folder, Task L).

Contents: PlacementMirror.exe (app.launcher) and _internal/ with Python, the app code, the
UI, the question bank, the Whisper assets, the Python packages and ONNX Runtime with the
QNN libraries. Models are not built in: extract models-v1.zip into
dist/PlacementMirror/_internal (python tools/fetch_models.py --dest dist/PlacementMirror/_internal).

After PyInstaller it:
- keeps only the PortAudio DLL sounddevice loads on this CPU. The ASIO builds (the
  Steinberg ASIO SDK has its own license) and the other CPUs' builds are removed.
- writes THIRD_PARTY_LICENSES.md for exactly the Python distributions PyInstaller bundled
  (tools/third_party_licenses.py) next to the exe, with LICENSE and README.txt.

Needs requirements.txt and requirements-build.txt installed in the running Python.

Usage: python tools/build_app.py [--dist dist] [--licenses-to-repo]
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = "PlacementMirror"
PORTAUDIO_DLL = {"ARM64": "libportaudioarm64.dll", "AMD64": "libportaudio64bit.dll"}
README = """Placement Mirror
================

Run PlacementMirror.exe. It starts the app on this computer and opens it in your browser
(Microsoft Edge or another Chromium browser). Everything runs on this computer. Nothing is
sent over the network. Close this window or press Ctrl+C in it to stop the app.

Session reports are saved in %LOCALAPPDATA%\\PlacementMirror\\sessions.
Licenses: LICENSE (Placement Mirror) and THIRD_PARTY_LICENSES.md (everything it ships).
"""


def pyinstaller(dist: Path, work: Path) -> None:
    import PyInstaller.__main__

    sep = os.pathsep
    PyInstaller.__main__.run([
        str(ROOT / "app" / "launcher.py"),
        "--name", NAME, "--onedir", "--console", "--noconfirm", "--clean",
        "--distpath", str(dist), "--workpath", str(work), "--specpath", str(work),
        "--paths", str(ROOT),
        "--add-data", f"{ROOT / 'app' / 'ui' / 'static'}{sep}app/ui/static",
        "--add-data", f"{ROOT / 'app' / 'audio' / 'assets'}{sep}app/audio/assets",
        "--add-data", f"{ROOT / 'questions' / 'bank.json'}{sep}questions",
        # ONNX Runtime's DLLs, and the QNN EP plugin with its QNN libraries and license files.
        "--collect-binaries", "onnxruntime", "--collect-data", "onnxruntime",
        "--collect-all", "onnxruntime_qnn",
        # uvicorn picks its protocol and loop modules by name at run time.
        "--collect-submodules", "uvicorn", "--hidden-import", "wsproto",
        "--collect-submodules", "app",
        # onnxruntime's optional tool packages pull in sympy and friends, the app does not use them.
        "--exclude-module", "onnxruntime.tools", "--exclude-module", "onnxruntime.transformers",
        "--exclude-module", "onnxruntime.quantization", "--exclude-module", "sympy",
        # Optional imports the app does not need: tkinter, pytest (via numpy.testing) and
        # readline (pyreadline3, via the standard library's interactive console modules).
        "--exclude-module", "tkinter", "--exclude-module", "pytest", "--exclude-module", "_pytest",
        "--exclude-module", "readline", "--exclude-module", "pyreadline3",
    ])


def prune_portaudio(internal: Path) -> list[str]:
    folder = internal / "_sounddevice_data" / "portaudio-binaries"
    keep = PORTAUDIO_DLL.get(platform.machine())
    if keep is None or not (folder / keep).exists():
        raise SystemExit(f"No PortAudio DLL for {platform.machine()} in {folder}")
    removed = []
    for f in folder.iterdir():
        if f.suffix.lower() in (".dll", ".dylib") and f.name != keep:
            f.unlink()
            removed.append(f.name)
    return removed


def bundled_modules(work: Path) -> set[str]:
    """Top level module names PyInstaller put in the app (its PYZ and binaries lists)."""
    names = set()
    for toc in (work / NAME).glob("*.toc"):
        text = toc.read_text(encoding="utf-8", errors="replace")
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                if len(item) == 3 and all(isinstance(x, str) for x in item) and item[2] in (
                        "PYMODULE", "EXTENSION", "PYSOURCE"):
                    names.add(item[0].split(".")[0])
                else:
                    stack.extend(item)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=ROOT / "dist")
    ap.add_argument("--licenses-to-repo", action="store_true",
                    help="also write THIRD_PARTY_LICENSES.md to the repository root")
    args = ap.parse_args()
    work = ROOT / "build" / "pyinstaller"
    pyinstaller(args.dist, work)
    app_dir = args.dist / NAME
    internal = app_dir / "_internal"
    removed = prune_portaudio(internal)
    print(f"Removed PortAudio builds not used on {platform.machine()}: {removed}")

    from tools.third_party_licenses import bundled_distributions, render

    dists = bundled_distributions(bundled_modules(work))
    text, referenced = render(dists)
    for rel, source in referenced:  # notices the file points to instead of copying must ship
        target = internal / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"Added {rel} to the app folder (referenced by THIRD_PARTY_LICENSES.md)")
    (app_dir / "THIRD_PARTY_LICENSES.md").write_text(text, encoding="utf-8", newline="\n")
    if args.licenses_to_repo:
        (ROOT / "THIRD_PARTY_LICENSES.md").write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "LICENSE", app_dir / "LICENSE")
    (app_dir / "README.txt").write_text(README, encoding="utf-8", newline="\r\n")
    print(f"Python distributions bundled: {', '.join(d.metadata['Name'] for d in dists)}")
    print(f"Built {app_dir}. Add the models: python tools/fetch_models.py --dest {internal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
