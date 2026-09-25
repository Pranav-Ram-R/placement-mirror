"""Download the app's models into models/ and write models/manifest.json.

AI Hub models: every compile job referenced in benchmarks/raw/jobs_*.json (compile
entries, compile_job_id and source_model_from_compile_job fields) is looked up. For each
app model and runtime (onnx and precompiled_qnn_onnx) the newest successful compile job
is downloaded to models/<name>/<runtime>/, keeping the file names AI Hub uses, because a
precompiled model refers to its context binary by relative path.

Silero VAD: the ONNX model from the official snakers4/silero-vad repository at a release
tag, with version and license recorded.

models/ is gitignored except models/manifest.json.

Usage: python -m aihub.fetch_models [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "benchmarks" / "raw"
MODELS = REPO / "models"
MANIFEST = MODELS / "manifest.json"

# app model name -> model name used in the job logs
AIHUB_MODELS = {
    "face_detector": "mediapipe_face_float_face_detector",
    "face_landmark": "mediapipe_face_float_face_landmark_detector",
    "pose_detector": "mediapipe_pose_float_pose_detector",
    "pose_landmark": "mediapipe_pose_float_pose_landmark_detector",
    "whisper_tiny_encoder": "whisper_tiny_float_encoder",
    "whisper_tiny_decoder": "whisper_tiny_float_decoder",
}
RUNTIMES = ("onnx", "precompiled_qnn_onnx")

SILERO_VERSION = "v6.2.3"
SILERO_URL = f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx"
SILERO_LICENSE = "MIT (https://github.com/snakers4/silero-vad/blob/master/LICENSE)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_list(folder: Path) -> list[dict]:
    return [{"path": p.relative_to(REPO).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(folder.iterdir()) if p.is_file()]


def referenced_compile_jobs() -> dict[str, str]:
    """compile job id -> model name, from every job log."""
    refs: dict[str, str] = {}
    for log in sorted(RAW.glob("jobs_*.json")):
        for e in json.loads(log.read_text(encoding="utf-8")):
            if e.get("job_type") == "compile" and e.get("job_id"):
                refs[e["job_id"]] = e["model"]
            for key in ("compile_job_id", "source_model_from_compile_job"):
                if e.get(key):
                    refs[e[key]] = e["model"]
    return refs


def unpack(download: Path, dest: Path) -> None:
    """Copy the downloaded model (a .onnx file or a zip holding model.onnx plus data) into dest."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if download.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(download).extractall(tmp)
            onnx_files = list(Path(tmp).rglob("*.onnx"))
            if len(onnx_files) != 1:
                raise SystemExit(f"{download}: expected one .onnx in the zip, found {onnx_files}")
            for f in onnx_files[0].parent.iterdir():
                shutil.copy2(f, dest / f.name)
    else:
        shutil.copy2(download, dest / download.name)


def epcontext_refs(onnx_path: Path) -> list[str]:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    refs = []
    for node in m.graph.node:
        if node.op_type == "EPContext":
            attrs = {a.name: a for a in node.attribute}
            if "embed_mode" in attrs and attrs["embed_mode"].i == 0 and "ep_cache_context" in attrs:
                refs.append(attrs["ep_cache_context"].s.decode())
    return refs


def fetch_aihub(client, force: bool, old: dict) -> list[dict]:
    from aihub.collect_results import option, precision_from

    candidates: dict[tuple[str, str], list] = {}
    for job_id, model in referenced_compile_jobs().items():
        app_name = next((k for k, v in AIHUB_MODELS.items() if v == model), None)
        if app_name is None:
            continue
        job = client.get_job(job_id)
        if not job.get_status().success:
            continue
        runtime = option(job.options, "--target_runtime")
        if runtime in RUNTIMES:
            candidates.setdefault((app_name, runtime), []).append(job)

    entries = []
    for app_name in AIHUB_MODELS:
        for runtime in RUNTIMES:
            jobs = sorted(candidates.get((app_name, runtime), []), key=lambda j: j.date, reverse=True)
            if not jobs:
                raise SystemExit(f"No successful {runtime} compile job for {app_name} in the job logs")
            job = jobs[0]
            dest = MODELS / app_name / runtime
            prev = old.get((app_name, runtime))
            if not force and prev and prev["job_id"] == job.job_id and all(
                    (REPO / f["path"]).exists() and sha256(REPO / f["path"]) == f["sha256"] for f in prev["files"]):
                print(f"  {app_name:22} {runtime:21} {job.job_id} already present")
                entries.append(prev)
                continue
            target = job.get_target_model()
            with tempfile.TemporaryDirectory() as tmp:
                download = Path(target.download(str(Path(tmp) / f"{app_name}_{runtime}")))
                unpack(download, dest)
            onnx_path = next(dest.glob("*.onnx"))
            missing = [r for r in epcontext_refs(onnx_path) if not (dest / r).exists()]
            if missing:
                raise SystemExit(f"{onnx_path}: EPContext refers to missing files {missing}")
            precision, why = precision_from(job, target)
            metadata = {str(k).split(".")[-1]: str(v) for k, v in (target.metadata or {}).items()}
            entries.append({
                "name": app_name, "runtime": runtime, "precision": precision, "precision_evidence": why,
                "job_id": job.job_id, "source": "AI Hub compile job", "aihub_model": AIHUB_MODELS[app_name],
                "compile_options": job.options, "target_model_id": target.model_id,
                "qairt_sdk_version": metadata.get("QAIRT_SDK_VERSION"),
                "onnx": onnx_path.relative_to(REPO).as_posix(), "sha256": sha256(onnx_path),
                "files": file_list(dest), "other_candidates": [j.job_id for j in jobs[1:]],
            })
            print(f"  {app_name:22} {runtime:21} {job.job_id} -> {dest.relative_to(REPO).as_posix()} ({precision})")
    return entries


def fetch_silero(force: bool, old: dict) -> dict:
    import onnx

    dest = MODELS / "silero_vad" / "onnx"
    path = dest / "silero_vad.onnx"
    prev = old.get(("silero_vad", "onnx"))
    if force or not (prev and path.exists() and sha256(path) == prev["sha256"] and prev["version"] == SILERO_VERSION):
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SILERO_URL, timeout=120) as r:
            path.write_bytes(r.read())
    m = onnx.load(str(path))
    io_types = sorted({onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()
                       for v in list(m.graph.input) + list(m.graph.output)})
    float_types = [t for t in io_types if t.startswith("float") or t in ("double",)]
    precision = "float32" if float_types == ["float"] else "unknown"
    print(f"  silero_vad             onnx                  {SILERO_VERSION} -> {dest.relative_to(REPO).as_posix()} ({precision})")
    return {
        "name": "silero_vad", "runtime": "onnx", "precision": precision,
        "precision_evidence": f"graph input and output types {io_types}",
        "job_id": None, "source": "snakers4/silero-vad GitHub release", "version": SILERO_VERSION,
        "license": SILERO_LICENSE, "url": SILERO_URL, "opset": [o.version for o in m.opset_import],
        "onnx": path.relative_to(REPO).as_posix(), "sha256": sha256(path), "files": file_list(dest),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download again even if files match the manifest")
    args = ap.parse_args()
    import qai_hub as hub

    old = {}
    if MANIFEST.exists():
        old = {(m["name"], m["runtime"]): m for m in json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]}
    print("AI Hub target models")
    models = fetch_aihub(hub.Client(), args.force, old)
    print("Silero VAD")
    models.append(fetch_silero(args.force, old))
    MANIFEST.write_text(json.dumps({
        "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": models,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(REPO).as_posix()} with {len(models)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
