"""THIRD_PARTY_LICENSES.md for the packaged app (Task L).

Covers what the app ships:
- the models (MediaPipe face and pose, Whisper tiny, Silero VAD), from models/manifest.json
  and the license texts in packaging/licenses/ (sources in packaging/licenses/SOURCES.md)
- native parts: the Python runtime, the QNN libraries in onnxruntime-qnn, PortAudio and
  the PyInstaller bootloader
- every Python distribution PyInstaller bundled, with the license files each one ships.
  tools/build_app.py passes the bundled set. Run on its own, this uses the packages in
  requirements.txt and their dependencies instead.

Text files over 100 KB are not copied in. They ship unchanged in the app folder and are
listed by their path there (tools/build_app.py makes sure they are in it).

Usage: python tools/third_party_licenses.py [--out THIRD_PARTY_LICENSES.md]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LICENSES = ROOT / "packaging" / "licenses"
MAX_INLINE = 100_000
NOTICE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "THIRDPARTYNOTICES", "THIRD_PARTY", "AUTHORS")

MODELS = [
    {"title": "MediaPipe face detector and face landmark models", "names": ("face_detector", "face_landmark"),
     "origin": "MediaPipe Face Detection and Face Mesh (Google), PyTorch conversion MediaPipePyTorch "
               "(Zak Murez), exported and compiled with Qualcomm AI Hub (qai_hub_models mediapipe_face).",
     "license": "Apache License 2.0",
     "evidence": "qai_hub_models 0.63.0 mediapipe_face manifest.yaml (license_type apache-2.0), "
                 "MediaPipePyTorch LICENSE (same terms as MediaPipe, Apache License 2.0).",
     "texts": ["MediaPipePyTorch-LICENSE.txt", "mediapipe-LICENSE.txt"]},
    {"title": "MediaPipe pose detector and pose landmark models", "names": ("pose_detector", "pose_landmark"),
     "origin": "MediaPipe Pose (BlazePose, Google), PyTorch conversion MediaPipePyTorch (Zak Murez), "
               "exported and compiled with Qualcomm AI Hub (qai_hub_models mediapipe_pose).",
     "license": "Apache License 2.0",
     "evidence": "qai_hub_models 0.63.0 mediapipe_pose manifest.yaml (license_type apache-2.0), "
                 "MediaPipePyTorch LICENSE.",
     "texts": ["MediaPipePyTorch-LICENSE.txt", "mediapipe-LICENSE.txt"]},
    {"title": "Whisper tiny encoder and decoder", "names": ("whisper_tiny_encoder", "whisper_tiny_decoder"),
     "origin": "OpenAI Whisper tiny (openai/whisper-tiny checkpoint) in the Hugging Face transformers "
               "implementation, exported and compiled with Qualcomm AI Hub (qai_hub_models whisper_tiny).",
     "license": "Apache License 2.0 (openai/whisper-tiny checkpoint and transformers), MIT (OpenAI Whisper "
                "repository)",
     "evidence": "qai_hub_models 0.63.0 whisper_tiny manifest.yaml (license_type apache-2.0), openai/whisper-tiny "
                 "model card (license: apache-2.0), openai/whisper LICENSE (MIT).",
     "texts": ["transformers-LICENSE.txt", "openai-whisper-LICENSE.txt"]},
    {"title": "Silero VAD", "names": ("silero_vad",),
     "origin": "silero_vad.onnx from the snakers4/silero-vad v6.2.3 release.",
     "license": "MIT", "evidence": "snakers4/silero-vad v6.2.3 LICENSE.",
     "texts": ["silero-vad-LICENSE.txt"]},
]


def fence(text: str) -> str:
    # Line endings normalized, so the file is the same whichever line endings a license file uses.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return f"````text\n{text.rstrip()}\n````\n"


def requirement_roots(path: Path = ROOT / "requirements.txt") -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(Requirement(line).name)
    return names


def dependency_closure(roots: list[str]) -> list[metadata.Distribution]:
    seen: dict[str, metadata.Distribution] = {}
    stack = list(roots)
    while stack:
        name = canonicalize_name(stack.pop())
        if name in seen:
            continue
        dist = metadata.distribution(name)
        seen[name] = dist
        for req in dist.requires or []:
            r = Requirement(req)
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                stack.append(r.name)
    return sorted(seen.values(), key=lambda d: d.metadata["Name"].lower())


def bundled_distributions(modules: set[str]) -> list[metadata.Distribution]:
    """Distributions that own the given top level modules (the ones PyInstaller bundled)."""
    owners = metadata.packages_distributions()
    names = {canonicalize_name(d) for m in modules for d in owners.get(m, [])}
    return sorted((metadata.distribution(n) for n in names), key=lambda d: d.metadata["Name"].lower())


def license_statement(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    lic = (meta.get("License") or "").strip()
    if lic and len(lic) <= 120 and "\n" not in lic:
        return lic
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return ", ".join(classifiers) or ("see the license text below" if lic else "not stated in the package metadata")


def license_files(dist: metadata.Distribution) -> list:
    """License and notice files in the dist-info folder and at the top of the package folders."""
    out = []
    for f in dist.files or []:
        parts = f.parts
        in_info = parts[0].endswith(".dist-info")
        named = f.name.upper().startswith(NOTICE_PREFIXES) or f.name.upper().startswith("QUALCOMM_LICENSE")
        if (in_info and (len(parts) > 2 and parts[1] == "licenses" or named)) or (not in_info and len(parts) == 2 and named):
            out.append(f)
    return sorted(out, key=str)


def home_page(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    if meta.get("Home-page"):
        return meta["Home-page"]
    for url in meta.get_all("Project-URL") or []:
        label, _, link = url.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "repository", "source code"):
            return link.strip()
    return ""


def render(dists: list[metadata.Distribution] | None = None) -> tuple[str, list[tuple[str, Path]]]:
    """The file text, and the package files it refers to instead of copying (large or binary),
    as (path in the app's _internal folder, source file)."""
    dists = dists if dists is not None else dependency_closure(requirement_roots())
    referenced: list[tuple[str, Path]] = []
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# Third party licenses",
        "",
        "Placement Mirror is MIT licensed (LICENSE). The packaged app also ships the models, native",
        "libraries and Python packages below, under their own licenses. Generated by",
        f"tools/third_party_licenses.py on {platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}. Paths starting with _internal/ are in the app folder.",
        "",
        "## Models",
        "",
        "The face, pose and Whisper model files are outputs of Qualcomm AI Hub compile jobs (job ids",
        "in models/manifest.json). qai_hub_models, the code that exported them, is BSD-3-Clause.",
        "Qualcomm AI Hub's own terms of service cover the compile service and are not reproduced here.",
        "",
    ]
    for m in MODELS:
        files = sorted({f["path"] for e in manifest["models"] if e["name"] in m["names"] for f in e["files"]})
        lines += [f"### {m['title']}", "", f"- Origin: {m['origin']}", f"- License: {m['license']}",
                  f"- License statement from: {m['evidence']}", f"- Files: {', '.join(files)}", ""]
    lines += ["### Model license texts", ""]
    for name in sorted({t for m in MODELS for t in m["texts"]}):
        lines += [f"#### {name}", "", fence((LICENSES / name).read_text(encoding="utf-8"))]

    lines += ["## Native libraries", ""]
    py_license = Path(sys.base_prefix) / "LICENSE.txt"
    lines += [f"### Python {platform.python_version()} runtime", "",
              "The Python interpreter and standard library (python3*.dll, _internal/base_library.zip and the",
              "standard library extension modules). Python Software Foundation License.", ""]
    lines += [fence(py_license.read_text(encoding="utf-8")) if py_license.exists() else
              "The Python LICENSE.txt was not found in this Python installation.\n"]
    lines += ["### Qualcomm AI Engine Direct (QNN) libraries", "",
              "In the onnxruntime-qnn package (_internal/onnxruntime_qnn/libs). Qualcomm Technologies, Inc.",
              "Terms and Conditions of Use, AI Stack License, shipped unchanged as",
              "_internal/onnxruntime_qnn/Qualcomm_LICENSE.pdf. It allows distribution in object code form",
              "as part of an application only, and does not allow removing its notices.", "",
              "### PortAudio", "",
              "In sounddevice (_internal/_sounddevice_data/portaudio-binaries). Only the DLL for the build's",
              "CPU, without ASIO, is shipped. MIT style license, Ross Bencina and Phil Burk.", "",
              fence((LICENSES / "portaudio-LICENSE.txt").read_text(encoding="utf-8"))]
    try:
        pyi = metadata.distribution("pyinstaller")
        lines += [f"### PyInstaller {pyi.version} bootloader", "",
                  "PlacementMirror.exe is the PyInstaller bootloader. GPL-2.0-or-later with the bootloader",
                  "exception, which allows distributing programs built with it under any license.", ""]
        for f in license_files(pyi):
            lines += [f"#### {f}", "", fence(Path(pyi.locate_file(f)).read_text(encoding="utf-8", errors="replace"))]
    except metadata.PackageNotFoundError:
        lines += ["### PyInstaller bootloader", "", "PyInstaller is not installed in this environment.", ""]

    lines += ["## Python packages", "", "| Package | Version | License |", "|---|---|---|"]
    for d in dists:
        lines.append(f"| {d.metadata['Name']} | {d.version} | {license_statement(d)} |")
    lines.append("")
    for d in dists:
        lines += [f"### {d.metadata['Name']} {d.version}", "", f"- License: {license_statement(d)}"]
        if home_page(d):
            lines.append(f"- Home: {home_page(d)}")
        lines.append("")
        files = license_files(d)
        if not files:
            lines += ["No license file in the package. The license statement above is from its metadata.", ""]
        for f in files:
            path = Path(d.locate_file(f))
            data = path.read_bytes()
            if f.suffix.lower() == ".pdf" or len(data) > MAX_INLINE:
                referenced.append((f.as_posix(), path))
                lines += [f"#### {f}", "", f"Not copied here ({len(data) / 1000:.0f} KB"
                          f"{', PDF' if f.suffix.lower() == '.pdf' else ''}). Shipped unchanged as _internal/{f}.", ""]
            else:
                lines += [f"#### {f}", "", fence(data.decode("utf-8", errors="replace"))]
    return "\n".join(lines).rstrip() + "\n", referenced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "THIRD_PARTY_LICENSES.md")
    args = ap.parse_args()
    text, _ = render()
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {args.out} ({len(text) / 1000:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
