"""Day 3 optimization experiment: whisper_tiny runtime format x precision on the NPU.

Logs every job to benchmarks/raw/jobs_day3.json (collected by aihub.collect_results into
benchmarks/raw/day3_aihub_results.json). All jobs target the hosted "Snapdragon X Elite CRD".

Cells, encoder and decoder each. One variable changes at a time, the rest of the compile
options are copied from the sibling cell's compile job:
  onnx_fp32         existing Day 1 compiles j5688od7g, jp3zzowz5
  precompiled_fp16  existing Day 0 compiles jp1nnvn2g, jp4yy9yvp
  onnx_fp16         NEW: onnx_fp32 options + --quantize_full_type float16 --quantize_io
  precompiled_fp32  NEW: precompiled_fp16 options without --quantize_full_type float16
                    --quantize_io, --qairt_version latest pinned to 2.50 (what "latest"
                    resolved to for the Day 0 compiles: 2.50.0.260828221209, models/manifest.json)
New compiles use the same source models and input specs as the existing ones (read from
their compile jobs). All eight cells are profiled in this batch with the same options
(PROFILE_OPTIONS), existing compiled models included, so profile settings do not differ.

Decisions (2026-09-27, author): the AI Hub docs say --quantize_full_type cannot be used
with target runtime ONNX and do not list float16 as a value. onnx_fp16 is submitted as
specified anyway and whatever AI Hub does is recorded. precompiled_fp32 is submitted as
specified (no quantize flags), although the docs say the HTP backend assumes a quantized
network when no precision is set.

qai_hub prints non ASCII progress characters while waiting, so run with PYTHONIOENCODING=utf-8
on a Windows console.

Usage: python -m aihub.submit_day3 [--force]
       python -m aihub.submit_day3 --resume   (wait for logged compiles that have no profile yet)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import qai_hub as hub

from aihub.submit_batch import Log

LOG = Path(__file__).resolve().parents[1] / "benchmarks" / "raw" / "jobs_day3.json"
DEVICE = "Snapdragon X Elite CRD"
QAIRT = "2.50"
PROFILE_OPTIONS = f"--compute_unit npu --qairt_version {QAIRT} --max_profiler_iterations 100"
FP16_FLAGS = "--quantize_full_type float16 --quantize_io"
GROUP = "Day 3 whisper_tiny runtime x precision on NPU"

EXISTING = {
    ("onnx_fp32", "encoder"): "j5688od7g",
    ("onnx_fp32", "decoder"): "jp3zzowz5",
    ("precompiled_fp16", "encoder"): "jp1nnvn2g",
    ("precompiled_fp16", "decoder"): "jp4yy9yvp",
}
# new cell -> (sibling cell its options come from, how)
NEW = {
    "onnx_fp16": ("onnx_fp32", "add --quantize_full_type float16 --quantize_io"),
    "precompiled_fp32": ("precompiled_fp16", "remove --quantize_full_type float16 --quantize_io, pin --qairt_version 2.50"),
}

# Quoted from https://app.aihub.qualcomm.com/docs/hub/api.html, fetched 2026-09-27.
DOCS = "https://app.aihub.qualcomm.com/docs/hub/api.html (fetched 2026-09-27)"
FLAG_DOCS = {
    "--target_runtime": "Overrides the default target runtime. [...] onnx : ONNX Runtime ( .onnx ) "
                        "precompiled_qnn_onnx : ONNX Runtime model with an embedded Qualcomm AI Engine Direct "
                        "context binary.",
    "--output_names": "Overrides the default output names. [...] When used, a name must be specified for each "
                      "model output.",
    "--qairt_version": "Specifies the version of Qualcomm AI Runtime to use in the job. This is applicable to: "
                       "[...] Compile jobs targeting ONNX Runtime with an embedded Qualcomm AI Engine Direct "
                       "context binary (i.e., precompiled_qnn_onnx ). All link, inference, and profile jobs. "
                       "[...] Valid arguments are: major.minor version [...] latest , which will select the "
                       "latest version.",
    "--qnn_options": "Specify behavior when generating a Qualcomm AI Engine Direct context binary. [...] "
                     "context_enable_graphs=<name> On- and offline preparation : The name of the graphs to "
                     "compile into a model file or load at runtime.",
    "--quantize_full_type": "Quantizes an unquantized model to the specified type. [...] Options: int8 [...] "
                            "int16 [...] w8a16 [...] w4a8 [...] w4a16 [...] Requirements: This option cannot be "
                            "used if the input is an AIMET model, or if the target runtime is ONNX. "
                            "NOTE: float16 is not a documented value and ONNX is a documented exclusion. "
                            "precompiled_fp16 (Day 0) used float16 and compiled. onnx_fp16 uses it with ONNX "
                            "by the author's decision.",
    "--quantize_io": "Quantize the input and outputs when quantizing a model. Options: true (or no argument) "
                     "false Default: Inputs and output are not quantized.",
    "--compute_unit": "Specifies the target compute unit(s). [...] When used in a profile or inference job, "
                      "this targets the specified compute unit(s). Implicit fall back to CPU will be determined "
                      "by the target_runtime ; TfLite and ONNX Runtime always include CPU fallback. [...] npu "
                      "Target NPU.",
    "--max_profiler_iterations": "Specifies the maximum number of profile iterations. Fewer iterations may be "
                                 "run if their cumulative runtime is predicted to exceed the execution timeout "
                                 "defined by --max_profiler_time . [...] Default: 100",
}
# Doc statements that bear on what the precision variable changes (quoted, not flags we set).
CONTEXT_DOCS = {
    "qnn_enable_htp_fp16_precision (ONNX Runtime QNN EP option, not set here)":
        "Enable a fp32 model to be inferenced with fp16 precision. This option greatly improves performance "
        "of models with any fp32 inputs. Default is 1 (enable).",
    "default_graph_htp_precision (QNN option, not set here)":
        "If no precision value is set, the QNN HTP backend assumes that the client expects to run a quantized "
        "network. When the precision value is set to FLOAT16 , the QNN HTP backend will convert user provided "
        "float32 inputs to float16 and execute the graph with float16 math.",
}


def flags_in(options: str) -> dict:
    used = re.findall(r"(--\w+)", options)
    return {f: FLAG_DOCS[f] for f in dict.fromkeys(used)}


def model_name(component: str) -> str:
    return f"whisper_tiny_float_{component}"


def new_options(cell: str, sibling_options: str) -> str:
    if cell == "onnx_fp16":
        if "--quantize" in sibling_options:
            raise SystemExit(f"onnx_fp32 options already quantize: {sibling_options}")
        return f"{sibling_options} {FP16_FLAGS}"
    if FP16_FLAGS not in sibling_options:
        raise SystemExit(f"precompiled_fp16 options lack {FP16_FLAGS!r}: {sibling_options}")
    options = " ".join(sibling_options.replace(FP16_FLAGS, "").split())
    return re.sub(r"--qairt_version\s+\S+", f"--qairt_version {QAIRT}", options)


def profile(client, log, target, component, cell, runtime, compile_job_id, device) -> None:
    entry = dict(job_type="profile", model=model_name(component), cell=cell, runtime=runtime, compute_unit="NPU",
                 options=PROFILE_OPTIONS, flags_checked=flags_in(PROFILE_OPTIONS), docs=DOCS,
                 compile_job_id=compile_job_id, group=GROUP, device=DEVICE)
    try:
        job = client.submit_profile_job(model=target, device=device, name=f"day3_{model_name(component)}_{cell}_npu",
                                        options=PROFILE_OPTIONS)
        log.add(job_id=job.job_id, url=job.url, **entry)
    except Exception as e:  # noqa: BLE001
        log.add(job_id=None, status="SUBMIT_FAILED", error=f"{type(e).__name__}: {e}", **entry)


def finish_compiles(client, log, pending, device) -> None:
    """Wait for each (compile job, component, cell, runtime), then profile it or log why not."""
    for job, component, cell, runtime in pending:
        status = job.wait()
        entry = next(e for e in log.entries if e.get("job_id") == job.job_id)
        entry["status"] = status.code
        if status.success:
            log.save()
            profile(client, log, job.get_target_model(), component, cell, runtime, job.job_id, device)
        else:
            entry["error"] = status.message
            log.save()
            log.add(job_type="profile", job_id=None, model=model_name(component), cell=cell, runtime=runtime,
                    compute_unit="NPU", options=PROFILE_OPTIONS, compile_job_id=job.job_id, group=GROUP,
                    device=DEVICE, status="NOT_SUBMITTED", error=f"compile job {job.job_id} failed: {status.message}")


def resume() -> int:
    client = hub.Client()
    device = hub.Device(DEVICE)
    log = Log(LOG)
    log.entries = json.loads(LOG.read_text(encoding="utf-8"))
    profiled = {e.get("compile_job_id") for e in log.entries if e["job_type"] == "profile"}
    pending = [(client.get_job(e["job_id"]), e["model"].rsplit("_", 1)[1], e["cell"], e["runtime"])
               for e in log.entries if e["job_type"] == "compile" and e.get("job_id") and e["job_id"] not in profiled]
    print(f"Resuming {len(pending)} compile jobs without a profile: {[j.job_id for j, *_ in pending]}", flush=True)
    finish_compiles(client, log, pending, device)
    n = sum(1 for e in log.entries if e["job_type"] == "profile" and e.get("job_id"))
    print(f"Done. {len(log.entries)} log entries, {n} profile jobs submitted. Log: {LOG}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="submit again even if the log exists")
    ap.add_argument("--resume", action="store_true", help="wait for logged compiles and profile them")
    args = ap.parse_args()
    if args.resume:
        return resume()
    if LOG.exists() and not args.force:
        print(f"{LOG} exists. Jobs were already submitted. Use --force to submit a second batch.")
        return 1
    client = hub.Client()
    device = hub.Device(DEVICE)
    log = Log(LOG)

    print("Profiling the existing onnx_fp32 and precompiled_fp16 compiled models", flush=True)
    for (cell, component), jid in EXISTING.items():
        cj = client.get_job(jid)
        runtime = re.search(r"--target_runtime\s+(\S+)", cj.options).group(1)
        profile(client, log, cj.get_target_model(), component, cell, runtime, jid, device)

    print("Submitting the new compiles", flush=True)
    pending = []
    for cell, (sibling, how) in NEW.items():
        for component in ("encoder", "decoder"):
            base = client.get_job(EXISTING[(sibling, component)])
            options = new_options(cell, base.options)
            runtime = re.search(r"--target_runtime\s+(\S+)", options).group(1)
            entry = dict(job_type="compile", model=model_name(component), cell=cell, runtime=runtime,
                         compute_unit=None, options=options, flags_checked=flags_in(options), docs=DOCS,
                         derived_from=f"{sibling} compile job {base.job_id}: {how}",
                         source_model_id=base.model.model_id, input_specs_from=base.job_id,
                         context_docs=CONTEXT_DOCS, group=GROUP, device=DEVICE)
            try:
                job = client.submit_compile_job(model=base.model, device=device, input_specs=base.shapes,
                                                name=f"day3_{model_name(component)}_{cell}", options=options)
                log.add(job_id=job.job_id, url=job.url, **entry)
                pending.append((job, component, cell, runtime))
            except Exception as e:  # noqa: BLE001
                log.add(job_id=None, status="SUBMIT_FAILED", error=f"{type(e).__name__}: {e}", **entry)

    print("Waiting for the new compiles, then profiling them", flush=True)
    finish_compiles(client, log, pending, device)
    n = sum(1 for e in log.entries if e["job_type"] == "profile" and e.get("job_id"))
    print(f"Done. {len(log.entries)} log entries, {n} profile jobs submitted. Log: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
