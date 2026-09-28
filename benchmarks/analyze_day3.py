"""Day 3 optimization analysis: derived comparisons, decoder layer times, decoder I/O bytes.

Inputs (all in benchmarks/raw):
- day3_aihub_results.json: p50 Measurements of the four profiled cells
- day3_decoder_layers.json: per-layer Measurements of j5qld8y7p and j57e8z9qp
  (aihub/collect_day3_details.py)
- day3_decoder_signatures.json: decoder input and output tensors of both precompiled cells

Outputs:
- benchmarks/raw/day3_derived.json: Derived records (format effect at float32, storage
  effect within precompiled, for encoder and decoder, and the decoder layer time sums and
  their fp32 - fp16 difference per layer group)
- benchmarks/raw/day3_decoder_io_bytes.json: bytes per decoder call from tensor shapes.
  These are not measurements and not Derived records (Derived needs Measurement inputs).
- the tables printed to stdout, which benchmarks/optimization.md quotes

Usage: python -m benchmarks.analyze_day3
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import prod
from pathlib import Path

from benchmarks.schema import Derived, Measurement, load_json, save_json

RAW = Path(__file__).resolve().parent / "raw"
PROFILES = {  # (cell, component) -> profile job
    ("onnx_fp32", "encoder"): "jpe7n9y75", ("onnx_fp32", "decoder"): "jgzl0enz5",
    ("precompiled_fp16", "encoder"): "jp1nm06kg", ("precompiled_fp16", "decoder"): "j57e8z9qp",
    ("precompiled_fp32", "encoder"): "jp0mxyj2g", ("precompiled_fp32", "decoder"): "j5qld8y7p",
}
# name -> (cell compared, baseline cell)
COMPARISONS = {
    "format_effect_at_fp32": ("precompiled_fp32", "onnx_fp32"),
    "storage_effect_within_precompiled": ("precompiled_fp16", "precompiled_fp32"),
}
DTYPE_BYTES = {"FLOAT": 4, "FLOAT16": 2, "INT32": 4, "INT64": 8}
TOP = 10
FP32_JOB, FP16_JOB = "j5qld8y7p", "j57e8z9qp"
LAYER_UNIT = "us (per-layer unit not documented, see notes)"


def p50s() -> dict[tuple[str, str], Measurement]:
    by_job = {r.job_id: r for r in load_json(RAW / "day3_aihub_results.json") if r.metric == "inference_time_p50"}
    return {key: by_job[job] for key, job in PROFILES.items()}


def derived(p50: dict) -> list[Derived]:
    out = []
    for name, (cell, base) in COMPARISONS.items():
        for comp in ("encoder", "decoder"):
            a, b = p50[(cell, comp)], p50[(base, comp)]
            out.append(Derived(
                name=f"whisper_tiny_{comp}_{name}_p50_difference", value=a.value - b.value, unit="us",
                formula=f"p50({cell} {comp}, {a.job_id}) - p50({base} {comp}, {b.job_id})", inputs=[a, b]))
            out.append(Derived(
                name=f"whisper_tiny_{comp}_{name}_p50_relative", value=(a.value - b.value) / b.value * 100,
                unit="%", formula=f"(p50({cell} {comp}, {a.job_id}) - p50({base} {comp}, {b.job_id})) "
                                  f"/ p50({base} {comp}, {b.job_id}) * 100", inputs=[a, b]))
    return out


def layer_groups(names: list[str]) -> dict[str, list[str]]:
    groups = {
        "input_output_layers": [n for n in names if n in ("Input", "Output")],
        "cross_cache_split_first_slices": [n for n in names if n in CROSS_SPLIT_SLICE0],
        "logits_layer": [n for n in names if n.startswith("logits")],
    }
    grouped = {n for g in groups.values() for n in g}
    groups["all_other_layers"] = [n for n in names if n not in grouped]
    return groups


def layer_derived(lay: dict[str, list[Measurement]]) -> list[Derived]:
    """Per-job layer sums and the fp32 - fp16 difference per layer group."""
    a = {layer_name(r): r for r in lay[FP32_JOB]}
    b = {layer_name(r): r for r in lay[FP16_JOB]}
    if list(a) != list(b):
        raise SystemExit("the two decoder profiles list different layers")
    out = [Derived(name=f"whisper_tiny_decoder_layer_time_sum_{job}", value=sum(r.value for r in lay[job]),
                   unit=LAYER_UNIT, formula=f"sum of execution_time over all {len(lay[job])} layers of {job}",
                   inputs=lay[job]) for job in (FP32_JOB, FP16_JOB)]
    for group, names in layer_groups(list(a)).items():
        for job, rows in ((FP32_JOB, a), (FP16_JOB, b)):
            out.append(Derived(
                name=f"whisper_tiny_decoder_layer_time_sum_{group}_{job}", value=sum(rows[n].value for n in names),
                unit=LAYER_UNIT, formula=f"sum of execution_time over {len(names)} layers ({group}) in {job}",
                inputs=[rows[n] for n in names]))
        out.append(Derived(
            name=f"whisper_tiny_decoder_layer_time_difference_{group}",
            value=sum(a[n].value for n in names) - sum(b[n].value for n in names), unit=LAYER_UNIT,
            formula=f"sum of execution_time over {len(names)} layers ({group}) in {FP32_JOB} minus the same "
                    f"layers in {FP16_JOB}",
            inputs=[a[n] for n in names] + [b[n] for n in names]))
    return out


def layers() -> dict[str, list[Measurement]]:
    out = defaultdict(list)
    for r in load_json(RAW / "day3_decoder_layers.json"):
        out[r.job_id].append(r)
    return out


def layer_name(r: Measurement) -> str:
    return r.metric.split(":", 1)[1]


def io_bytes() -> dict:
    sig = json.loads((RAW / "day3_decoder_signatures.json").read_text(encoding="utf-8"))["cells"]
    out = {"note": "Derived from tensor shapes and dtypes in day3_decoder_signatures.json, not a measurement. "
                   "bytes = product(shape) * bytes per element (FLOAT 4, FLOAT16 2, INT32 4), summed over "
                   "the graph inputs (passed in) and outputs (passed out) of one decoder call.",
           "cells": {}}
    for cell, s in sig.items():
        groups = defaultdict(int)
        for direction in ("inputs", "outputs"):
            for t in s[direction]:
                n = prod(t["shape"]) * DTYPE_BYTES[t["dtype"]]
                kind = ("cross_cache" if "cross" in t["name"] else "self_cache" if "self" in t["name"]
                        else t["name"])
                groups[f"{direction}:{kind}"] += n
        total_in = sum(v for k, v in groups.items() if k.startswith("inputs:"))
        total_out = sum(v for k, v in groups.items() if k.startswith("outputs:"))
        out["cells"][cell] = {"compile_job": s["compile_job"], "by_group": dict(groups), "bytes_in": total_in,
                              "bytes_out": total_out, "bytes_total": total_in + total_out}
    return out


def main() -> int:
    p50 = p50s()
    lay = layers()
    records = derived(p50) + layer_derived(lay)
    save_json(records, RAW / "day3_derived.json")
    print("Derived (benchmarks/raw/day3_derived.json)")
    for d in records:
        print(f"  {d.name} = {d.value:.2f} {d.unit}  [{d.formula}]")

    for job in (FP32_JOB, FP16_JOB):
        rows = lay[job]
        print(f"\nTop {TOP} layers by execution_time, {job} ({len(rows)} layers)")
        for r in sorted(rows, key=lambda r: -r.value)[:TOP]:
            print(f"  {r.value:>6.0f}  {layer_name(r)}")
    a = {layer_name(r): r.value for r in lay[FP32_JOB]}
    b = {layer_name(r): r.value for r in lay[FP16_JOB]}
    print("\nLayer groups: layers, fp32 sum, fp16 sum")
    for g, names in layer_groups(list(a)).items():
        print(f"  {g}: {len(names)}, {sum(a[n] for n in names):.0f}, {sum(b[n] for n in names):.0f}")

    ib = io_bytes()
    (RAW / "day3_decoder_io_bytes.json").write_text(json.dumps(ib, indent=1) + "\n", encoding="utf-8")
    print("\nDecoder bytes per call from tensor shapes (not a measurement)")
    for cell, c in ib["cells"].items():
        print(f"  {cell}: in {c['bytes_in']}, out {c['bytes_out']}, total {c['bytes_total']}  {c['by_group']}")
    return 0


# Split nodes on the cross attention cache inputs, from the source graph (the onnx float32
# decoder of compile job jp3zzowz5 has node_Split_<n> with input k_cache_cross_<i> or
# v_cache_cross_<i>). The precompiled graphs use the same layer names.
CROSS_SPLIT_SLICE0 = {f"node_Split_{n}_to_slice_0" for n in (55, 67, 155, 167, 237, 246, 337, 349)}

if __name__ == "__main__":
    sys.exit(main())
