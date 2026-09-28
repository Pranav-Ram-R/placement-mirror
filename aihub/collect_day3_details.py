"""Per-layer times and decoder signatures for the Day 3 optimization analysis.

Per-layer times: for the two precompiled decoder profile jobs (j5qld8y7p float32 I/O,
j57e8z9qp float16 I/O) every layer in execution_detail that has an execution_time becomes
a Measurement (Source.AIHUB_X_ELITE) in benchmarks/raw/day3_decoder_layers.json.

What these numbers are, from the sources checked on 2026-09-28:
- qai_hub 0.55.0 public_api_pb2.pyi, LayerDetail.execution_time: "If available, the
  minimum execution time for this layer." The unit is not stated there.
  ProfileJob.estimated_inference_time is documented as "Time spent in inference, in
  microseconds."
- The profile job logs show the layer times come from a separate task ("performing
  inference by layer", "Successfully ran model for 2 iterations") with the QNN EP option
  "profiling_level = optrace". The timed task (100 iterations) has no profiling_level line.
- ONNX Runtime QNN EP docs, profiling_level: "'off' - default. 'basic' 'detailed'
  'optrace' - Requires QAIRT 2.39 or later".

Decoder signatures: the target models of compile jobs jpxlzvxjp (precompiled float32)
and jp4yy9yvp (precompiled float16) are downloaded to a temporary folder and the input
and output names, dtypes and shapes of their ONNX wrappers are written to
benchmarks/raw/day3_decoder_signatures.json. Model files are not kept.

Usage: PYTHONIOENCODING=utf-8 python -m aihub.collect_day3_details
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import qai_hub as hub

from benchmarks.schema import Measurement, Source, load_json, save_json

RAW = Path(__file__).resolve().parents[1] / "benchmarks" / "raw"
PROFILE_JOBS = {"j5qld8y7p": "precompiled_fp32", "j57e8z9qp": "precompiled_fp16"}
COMPILE_JOBS = {"jpxlzvxjp": "precompiled_fp32", "jp4yy9yvp": "precompiled_fp16"}
LAYER_UNIT = "us (per-layer unit not documented, see notes)"
LAYER_NOTE = ("execution_detail[].execution_time from job.download_profile(). qai_hub 0.55.0 "
              "LayerDetail.execution_time: 'If available, the minimum execution time for this layer.' "
              "Taken in the profile job's separate 'performing inference by layer' task (2 iterations, "
              "QNN EP profiling_level = optrace per the job log), not in the 100-iteration timed task.")


def layer_records(client) -> list[Measurement]:
    p50 = {r.job_id: r for r in load_json(RAW / "day3_aihub_results.json")
           if r.metric == "inference_time_p50" and r.job_id in PROFILE_JOBS}
    out = []
    for job_id, cell in PROFILE_JOBS.items():
        base = p50[job_id]
        detail = client.get_job(job_id).download_profile()["execution_detail"]
        for i, d in enumerate(detail):
            if "execution_time" not in d:
                continue
            out.append(Measurement(
                model=base.model, metric=f"layer_execution_time:{d['name']}", value=d["execution_time"],
                unit=LAYER_UNIT, source=Source.AIHUB_X_ELITE, runtime=base.runtime, compute_unit=d["compute_unit"],
                precision=base.precision, exec_precision=base.exec_precision, qnn_version=base.qnn_version,
                ort_version=base.ort_version, job_id=job_id, timestamp=base.timestamp,
                notes=(f"{cell} | layer {i} of {len(detail)}, type {d['type']!r}, "
                       f"execution_cycles {d.get('execution_cycles')} | {LAYER_NOTE}"),
            ))
    return out


def tensor_specs(values) -> list[dict]:
    import onnx

    return [{"name": v.name, "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
             "shape": [x.dim_value if x.HasField("dim_value") else x.dim_param for x in v.type.tensor_type.shape.dim]}
            for v in values]


def signatures(client) -> dict:
    import onnx

    out = {}
    for job_id, cell in COMPILE_JOBS.items():
        target = client.get_job(job_id).get_target_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(target.download(str(Path(tmp) / "target")))
            if path.suffix == ".zip":
                zipfile.ZipFile(path).extractall(tmp)
            onnx_path = next(Path(tmp).rglob("*.onnx"))
            m = onnx.load(str(onnx_path), load_external_data=False)
        out[cell] = {"compile_job": job_id, "target_model_id": target.model_id,
                     "inputs": tensor_specs(m.graph.input), "outputs": tensor_specs(m.graph.output)}
    return out


def main() -> int:
    client = hub.Client()
    records = layer_records(client)
    save_json(records, RAW / "day3_decoder_layers.json")
    print(f"Wrote {len(records)} layer Measurements to benchmarks/raw/day3_decoder_layers.json")
    sig = signatures(client)
    (RAW / "day3_decoder_signatures.json").write_text(json.dumps({
        "source": "ONNX wrapper graph inputs and outputs of the AI Hub target models (not a measurement)",
        "cells": sig}, indent=1) + "\n", encoding="utf-8")
    print("Wrote benchmarks/raw/day3_decoder_signatures.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
