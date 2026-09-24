"""Submit the Day 1 AI Hub compile and profile jobs and log them to benchmarks/raw/jobs_day1.json.

All jobs target the hosted "Snapdragon X Elite CRD".

1. mediapipe_face detector and landmark compiled as precompiled_qnn_onnx, profiled on NPU.
2. All 4 vision models as runtime onnx, profiled with compute unit CPU and with GPU.
   Face reuses the Day 0 onnx target models. Pose is compiled to onnx here.
3. whisper_tiny encoder and decoder compiled as onnx float32, each profiled on CPU, GPU and NPU.

Source models are the ones uploaded by the Day 0 qai_hub_models export, read from the
Day 0 compile jobs. Input specs and compile options come from qai_hub_models so the
compiles match the model zoo recipe. QAIRT is pinned to the version the Day 0 jobs used.

Usage: python -m aihub.submit_batch [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib
import json
import re
import sys
from pathlib import Path

import qai_hub as hub

LOG = Path(__file__).resolve().parents[1] / "benchmarks" / "raw" / "jobs_day1.json"
DEVICE = "Snapdragon X Elite CRD"
QAIRT = "2.50"

# Day 0 compile jobs whose source models (qai_hub_models 0.63.0 export) are reused.
DAY0_COMPILE = {
    ("mediapipe_face", "face_detector"): "jgnzz7nrg",
    ("mediapipe_face", "face_landmark_detector"): "jprlln09p",
    ("mediapipe_pose", "pose_detector"): "jgnzz71kg",
    ("mediapipe_pose", "pose_landmark_detector"): "jprllnx0p",
    ("whisper_tiny", "encoder"): "jp1nnvn2g",
    ("whisper_tiny", "decoder"): "jp4yy9yvp",
}

# (zoo model, component, runtime to compile, compute units to profile, group)
COMPILES = [
    ("mediapipe_face", "face_detector", "precompiled_qnn_onnx", ["npu"], "1 shipping runtime NPU"),
    ("mediapipe_face", "face_landmark_detector", "precompiled_qnn_onnx", ["npu"], "1 shipping runtime NPU"),
    ("mediapipe_pose", "pose_detector", "onnx", ["cpu", "gpu"], "2 vision onnx CPU and GPU"),
    ("mediapipe_pose", "pose_landmark_detector", "onnx", ["cpu", "gpu"], "2 vision onnx CPU and GPU"),
    ("whisper_tiny", "encoder", "onnx", ["cpu", "gpu", "npu"], "3 whisper_tiny onnx float32"),
    ("whisper_tiny", "decoder", "onnx", ["cpu", "gpu", "npu"], "3 whisper_tiny onnx float32"),
]

# Existing Day 0 onnx target models profiled again on other compute units.
REUSE_TARGETS = [
    ("mediapipe_face", "face_detector", ["cpu", "gpu"], "2 vision onnx CPU and GPU"),
    ("mediapipe_face", "face_landmark_detector", ["cpu", "gpu"], "2 vision onnx CPU and GPU"),
]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def model_name(zoo_id: str, component: str) -> str:
    return f"{zoo_id}_float_{component}"


def pin_qairt(options: str) -> str:
    return re.sub(r"--qairt_version\s+\S+", f"--qairt_version {QAIRT}", options)


def profile_options(compute_unit: str) -> str:
    return f"--compute_unit {compute_unit} --qairt_version {QAIRT}"


@functools.cache
def zoo_model(zoo_id: str):
    from qai_hub_models.utils.asset_loaders import always_answer_prompts

    with always_answer_prompts(True):
        module = importlib.import_module(f"qai_hub_models.models.{zoo_id}")
        return module.Model.from_pretrained()


def zoo_recipe(zoo_id: str, component: str, runtime: str, device: hub.Device):
    """Input spec and compile options exactly as qai_hub_models would generate them."""
    from qai_hub_models import Precision, TargetRuntime
    from qai_hub_models.utils.input_spec import to_hub_input_specs

    model = zoo_model(zoo_id)
    spec = to_hub_input_specs(model.get_input_spec()[component])
    options = model.get_component_hub_compile_options(
        component, TargetRuntime(runtime), Precision.float, "", device
    )
    return spec, options


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []

    def add(self, **entry) -> dict:
        entry.setdefault("submitted_at", now())
        self.entries.append(entry)
        self.save()
        print(f"  {entry.get('job_type')} {entry.get('job_id')} {entry.get('model')} "
              f"{entry.get('runtime')} {entry.get('compute_unit')} {entry.get('status', '')}", flush=True)
        return entry

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")


def submit_profiles(client, log, target, name, runtime, units, group, compile_job_id, device):
    for unit in units:
        opts = profile_options(unit)
        entry = dict(job_type="profile", model=name, runtime=runtime, compute_unit=unit.upper(),
                     options=opts, compile_job_id=compile_job_id, group=group, device=DEVICE)
        try:
            job = client.submit_profile_job(model=target, device=device, name=f"day1_{name}_{runtime}_{unit}",
                                            options=opts)
            log.add(job_id=job.job_id, url=job.url, **entry)
        except Exception as e:
            log.add(job_id=None, status="SUBMIT_FAILED", error=f"{type(e).__name__}: {e}", **entry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="submit again even if the log exists")
    args = ap.parse_args()
    if LOG.exists() and not args.force:
        print(f"{LOG} exists. Jobs were already submitted. Use --force to submit a second batch.")
        return 1

    client = hub.Client()
    device = hub.Device(DEVICE)
    log = Log(LOG)

    print("Profiling Day 0 onnx target models on CPU and GPU", flush=True)
    for zoo_id, component, units, group in REUSE_TARGETS:
        cj = client.get_job(DAY0_COMPILE[(zoo_id, component)])
        runtime = re.search(r"--target_runtime\s+(\S+)", cj.options).group(1)
        submit_profiles(client, log, cj.get_target_model(), model_name(zoo_id, component), runtime,
                        units, group, cj.job_id, device)

    print("Submitting compile jobs", flush=True)
    pending = []
    for zoo_id, component, runtime, units, group in COMPILES:
        name = model_name(zoo_id, component)
        day0 = client.get_job(DAY0_COMPILE[(zoo_id, component)])
        spec, options = zoo_recipe(zoo_id, component, runtime, device)
        options = pin_qairt(options)
        day0_names = [n for n, _ in day0.shapes.items()] if isinstance(day0.shapes, dict) else None
        if day0_names is not None and sorted(day0_names) != sorted(spec):
            raise SystemExit(f"{name}: zoo input names {list(spec)} differ from Day 0 source {day0_names}")
        entry = dict(job_type="compile", model=name, runtime=runtime, compute_unit=None, options=options,
                     source_model_from_compile_job=day0.job_id, group=group, device=DEVICE)
        try:
            job = client.submit_compile_job(model=day0.model, device=device, name=f"day1_{name}_{runtime}",
                                            input_specs=spec, options=options)
            log.add(job_id=job.job_id, url=job.url, **entry)
            pending.append((job, name, runtime, units, group))
        except Exception as e:
            log.add(job_id=None, status="SUBMIT_FAILED", error=f"{type(e).__name__}: {e}", **entry)
            for unit in units:
                log.add(job_type="profile", job_id=None, model=name, runtime=runtime, compute_unit=unit.upper(),
                        options=profile_options(unit), group=group, device=DEVICE, status="NOT_SUBMITTED",
                        error=f"compile submit failed: {type(e).__name__}: {e}")

    print("Waiting for compile jobs, then submitting their profile jobs", flush=True)
    for job, name, runtime, units, group in pending:
        status = job.wait()
        entry = next(e for e in log.entries if e.get("job_id") == job.job_id)
        entry["status"] = status.code
        if status.success:
            log.save()
            submit_profiles(client, log, job.get_target_model(), name, runtime, units, group, job.job_id, device)
        else:
            entry["error"] = status.message
            log.save()
            for unit in units:
                log.add(job_type="profile", job_id=None, model=name, runtime=runtime, compute_unit=unit.upper(),
                        options=profile_options(unit), compile_job_id=job.job_id, group=group, device=DEVICE,
                        status="NOT_SUBMITTED", error=f"compile job {job.job_id} failed: {status.message}")

    n_profiles = sum(1 for e in log.entries if e["job_type"] == "profile" and e.get("job_id"))
    print(f"Done. {len(log.entries)} log entries, {n_profiles} profile jobs submitted. Log: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
