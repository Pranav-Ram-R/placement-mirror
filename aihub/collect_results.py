"""Collect finished Day 1 AI Hub profile jobs into Measurement records.

Reads benchmarks/raw/jobs_day1.json. For every finished profile job it saves the raw
per-iteration times to benchmarks/raw/samples/<job_id>.json and writes Measurement
records (Source.AIHUB_X_ELITE) to benchmarks/raw/day1_aihub_results.json. The output
file is rebuilt from AI Hub on every run, so reruns are safe. Unfinished jobs are
skipped. Failed jobs are printed with their error and their status is written back
to the job log.

Usage: python -m aihub.collect_results
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import qai_hub as hub

from benchmarks.schema import Measurement, Source, save_json

RAW = Path(__file__).resolve().parents[1] / "benchmarks" / "raw"
LOG = RAW / "jobs_day1.json"
OUT = RAW / "day1_aihub_results.json"
SAMPLES = RAW / "samples"
FP16_LINE = "enable_htp_fp16_precision = 1"


def option(options: str, name: str) -> str | None:
    toks = shlex.split(options or "")
    for i, tok in enumerate(toks):
        if tok == name and i + 1 < len(toks):
            return toks[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def precision_from(compile_job, target_model) -> tuple[str, str]:
    """Model precision from compile options and target model I/O dtypes only."""
    qtype = option(compile_job.options, "--quantize_full_type")
    io = [t[1] for spec in (target_model.input_spec, target_model.output_spec)
          for tensors in (spec or {}).values() for t in tensors]
    float_io = sorted({d for d in io if d.startswith("float")})
    if qtype == "float16" and float_io == ["float16"]:
        return "float16", (f"compile job {compile_job.job_id} has --quantize_full_type float16 "
                           "and target model float I/O is float16")
    if qtype is None and "--quantize" not in (compile_job.options or "") and float_io == ["float32"]:
        return "float32", (f"compile job {compile_job.job_id} has no quantize option "
                           "and target model float I/O is float32")
    return "unknown", f"compile options {compile_job.options!r} and float I/O {float_io} do not identify one precision"


def log_facts(job, logdir: Path) -> dict:
    paths = job.download_job_logs(str(logdir / job.job_id))
    paths = paths if isinstance(paths, list) else [paths]
    text = "".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in paths)
    eps = sorted(set(re.findall(r"Adding (\w+) Execution Provider", text)))
    qnn = sorted(set(re.findall(r"qnn_sdk.default.(\d+\.\d+\.\d+)", text)))
    ort = sorted(set(re.findall(r"ONNX Runtime version (\d+\.\d+\.\d+)", text)))
    fp16 = [re.sub(r"^\[[^\]]*\]\s*", "", l).strip() for l in text.splitlines() if FP16_LINE in l]
    return {
        "execution_providers": eps,
        # QNN version only when the job actually added the QNN execution provider.
        "qnn_version": qnn[0] if len(qnn) == 1 and "QNN" in eps else None,
        "ort_version": ort[0] if len(ort) == 1 else None,
        "fp16_line": fp16[0] if fp16 else None,
    }


def records_for(job, entry: dict, logdir: Path) -> list[Measurement]:
    profile = job.download_profile()
    # The AI Hub docs ("Working with Jobs", "Profile Jobs") show download_profile() as a flat
    # dict with 'all_inference_times' at the top level. With qai-hub 0.55.0 (checked 2026-09-25)
    # the observed structure is {'execution_summary': {..., 'estimated_inference_time',
    # 'inference_memory_peak_range', 'all_inference_times': [us, ...]}, 'execution_detail':
    # [{'name', 'type', 'compute_unit', 'execution_time', 'execution_cycles'}, ...]}.
    # The docs' execution_summary['execution_time'] key does not exist in these results.
    summary = profile["execution_summary"]
    samples = [float(x) for x in summary["all_inference_times"]]
    sample_path = SAMPLES / f"{job.job_id}.json"
    rel = sample_path.relative_to(RAW.parents[1]).as_posix()
    sample_path.write_text(json.dumps({
        "job_id": job.job_id, "model": entry["model"], "source": Source.AIHUB_X_ELITE.value, "device": entry["device"],
        "runtime": entry["runtime"], "requested_compute_unit": entry["compute_unit"],
        "field": "execution_summary.all_inference_times", "unit": "us", "count": len(samples), "samples": samples,
    }, indent=1) + "\n", encoding="utf-8")

    target = job.model
    compile_job = target.get_producer()
    runtime = option(compile_job.options, "--target_runtime")
    precision, precision_why = precision_from(compile_job, target)
    facts = log_facts(job, logdir)
    units = Counter(d["compute_unit"] for d in profile["execution_detail"])

    requested = option(job.options, "--compute_unit")
    notes = [
        job.url,
        (f"requested --compute_unit {requested}" if requested else "no --compute_unit option (AI Hub default)")
        + f" (profile options: {job.options})",
        "ops by compute unit: " + ", ".join(f"{u} {n}" for u, n in sorted(units.items())),
        f"runtime from --target_runtime of compile job {compile_job.job_id}",
        f"precision {precision}: {precision_why}",
        f"execution providers added in profile log: {', '.join(facts['execution_providers']) or 'none found'}",
        "ort_version from profile job log 'ONNX Runtime version' line",
        ("qnn_version from QNN SDK DLL directory in profile job log" if facts["qnn_version"]
         else "qnn_version None because the profile log shows no QNN execution provider"),
    ]
    if facts["fp16_line"]:
        notes.append(f"exec_precision float16 from profile job log line: {facts['fp16_line']}")
    common = dict(
        model=entry["model"], source=Source.AIHUB_X_ELITE, runtime=runtime,
        compute_unit="+".join(sorted(units)), precision=precision,
        exec_precision="float16" if facts["fp16_line"] else None,
        qnn_version=facts["qnn_version"], ort_version=facts["ort_version"],
        job_id=job.job_id, timestamp=job.date.isoformat(timespec="seconds"),
    )
    base = " | ".join(notes)
    n = len(samples)
    arr = np.asarray(samples)
    out = []
    for q in (50, 95):
        out.append(Measurement(
            metric=f"inference_time_p{q}", value=float(np.percentile(arr, q, method="linear")), unit="us",
            notes=f"{base} | p{q} = numpy.percentile(samples, {q}, method='linear') over the {n} per-iteration "
                  f"times (execution_summary.all_inference_times) in {rel}", **common))
    for stat, value in (("min", arr.min()), ("max", arr.max())):
        out.append(Measurement(
            metric=f"inference_time_{stat}", value=float(value), unit="us",
            notes=f"{base} | {stat} of the {n} per-iteration times in {rel}", **common))
    lo, hi = summary["inference_memory_peak_range"]
    out.append(Measurement(
        metric="inference_memory_peak_max", value=hi, unit="bytes",
        notes=f"{base} | upper end of execution_summary.inference_memory_peak_range ({lo} to {hi})", **common))
    for unit, count in sorted(units.items()):
        out.append(Measurement(metric=f"ops_on_{unit}", value=count, unit="ops", notes=base, **common))
    return out


def main() -> int:
    if not LOG.exists():
        print(f"{LOG} not found. Run aihub.submit_batch first.")
        return 1
    entries = json.loads(LOG.read_text(encoding="utf-8"))
    client = hub.Client()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    logdir = Path(tempfile.mkdtemp(prefix="aihub_logs_"))

    records: list[Measurement] = []
    finished, pending, failed = [], [], []
    for entry in entries:
        if not entry.get("job_id"):
            failed.append((entry, entry.get("error", "no job submitted")))
            continue
        job = client.get_job(entry["job_id"])
        status = job.get_status()
        entry["status"] = status.code
        if status.finished and not status.success:
            entry["error"] = status.message
            failed.append((entry, status.message))
        elif not status.finished:
            pending.append(entry)
        else:
            finished.append(entry)
            if entry["job_type"] == "profile":
                records += records_for(job, entry, logdir)
    LOG.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    save_json(records, OUT)

    def line(e):
        return f"{e.get('job_type', ''):8} {str(e.get('job_id')):10} {e['model']:45} {e['runtime']:21} {str(e.get('compute_unit') or '-'):4}"

    print(f"Finished ({len(finished)}):")
    for e in finished:
        print("  " + line(e))
    print(f"Pending ({len(pending)}):")
    for e in pending:
        print("  " + line(e) + f" {e['status']}")
    print(f"Failed ({len(failed)}):")
    for e, err in failed:
        print("  " + line(e) + f"\n      error: {err}")
    print(f"Wrote {len(records)} measurements to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
