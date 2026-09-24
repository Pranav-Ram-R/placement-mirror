"""Print the benchmark table from every record file in benchmarks/raw/.

One row per measured job with p50 and p95 latency. AI Hub's estimated_inference_time
is the minimum iteration (Day 0 finding), so it is not shown. Jobs without p50/p95
records are listed under the table. Failed jobs from the job logs (jobs_*.json) are
listed in a failures section with their exact error. They have no Measurement.

Usage: python -m benchmarks.table
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from benchmarks.schema import Measurement, load_json

RAW = Path(__file__).parent / "raw"
NOTE = "AI Hub estimated_inference_time is the minimum iteration. Tables report p50/p95."
FAILED_STATUSES = {"FAILED", "SUBMIT_FAILED", "NOT_SUBMITTED"}

COLUMNS = [
    ("model", "model"),
    ("source", "source"),
    ("runtime", "runtime"),
    ("precision", "precision"),
    ("exec_precision", "exec_precision"),
    ("compute unit", "compute_unit"),
    ("p50", "inference_time_p50"),
    ("p95", "inference_time_p95"),
    ("peak memory", "inference_memory_peak_max"),
    ("NPU ops", "ops_on_NPU"),
    ("GPU ops", "ops_on_GPU"),
    ("CPU ops", "ops_on_CPU"),
    ("qnn_version", "qnn_version"),
    ("ort_version", "ort_version"),
    ("job_id", "job_id"),
]


def _num(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.1f}"


def load_all(raw: Path = RAW) -> list[Measurement]:
    records = []
    for path in sorted(raw.glob("*.json")):
        if path.name.startswith("jobs_"):
            continue
        records += [r for r in load_json(path) if isinstance(r, Measurement)]
    return records


def load_failures(raw: Path = RAW) -> list[dict]:
    failures = []
    for path in sorted(raw.glob("jobs_*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if entry.get("status") in FAILED_STATUSES:
                failures.append({**entry, "log": path.name})
    return failures


def rows(records: list[Measurement]) -> list[dict[str, str]]:
    groups: dict[tuple, list[Measurement]] = defaultdict(list)
    for m in records:
        groups[(m.job_id, m.model, m.source, m.runtime, m.precision, m.exec_precision)].append(m)
    out = []
    for group in groups.values():
        first = group[0]
        metrics = {m.metric: f"{_num(m.value)} {m.unit}" for m in group}
        row = {
            "model": first.model,
            "source": first.source.value,
            "runtime": first.runtime,
            "precision": first.precision,
            "exec_precision": first.exec_precision or "None",
            "compute_unit": first.compute_unit,
            "qnn_version": first.qnn_version or "None",
            "ort_version": first.ort_version or "None",
            "job_id": first.job_id or "None",
        }
        for _, key in COLUMNS:
            row.setdefault(key, metrics.get(key, "-"))
        out.append(row)
    return sorted(out, key=lambda r: (r["model"], r["runtime"], r["compute_unit"], r["job_id"]))


def main() -> None:
    table = rows(load_all())
    shown = [r for r in table if r["inference_time_p50"] != "-" and r["inference_time_p95"] != "-"]
    omitted = [r for r in table if r not in shown]
    print(NOTE + "\n")
    headers = [h for h, _ in COLUMNS]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in shown:
        print("| " + " | ".join(row[key] for _, key in COLUMNS) + " |")
    unknown = [r for r in shown if r["precision"] == "unknown"]
    print(f"\n{len(shown)} jobs with p50/p95, {len(unknown)} with precision unknown")
    if omitted:
        print(f"Omitted, no p50/p95 records: " + ", ".join(f"{r['model']} {r['runtime']} ({r['job_id']})" for r in omitted))

    failures = load_failures()
    print(f"\nFailures ({len(failures)}, no Measurement):")
    for f in failures:
        print(f"- {f['model']} | runtime {f['runtime']} | compute unit {f.get('compute_unit') or '-'} | "
              f"job {f.get('job_id')} | {f['status']} | {f['log']}\n  error: {(f.get('error') or 'no error text').strip()}")


if __name__ == "__main__":
    main()
