"""Print one row per measured job from every record file in benchmarks/raw/.

Usage: python -m benchmarks.table
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmarks.schema import Measurement, load_json

RAW = Path(__file__).parent / "raw"

COLUMNS = [
    ("model", "model"),
    ("source", "source"),
    ("runtime", "runtime"),
    ("precision", "precision"),
    ("exec_precision", "exec_precision"),
    ("compute unit", "compute_unit"),
    ("latency (AI Hub estimate, = min)", "estimated_inference_time"),
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
    return f"{value:,.0f}" if value.is_integer() else f"{value:g}"


def load_all(raw: Path = RAW) -> list[Measurement]:
    records = []
    for path in sorted(raw.glob("*.json")):
        if path.name.startswith("jobs_"):
            continue
        records += [r for r in load_json(path) if isinstance(r, Measurement)]
    return records


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
    return out


def main() -> None:
    table = rows(load_all())
    headers = [h for h, _ in COLUMNS]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in table:
        print("| " + " | ".join(row[key] for _, key in COLUMNS) + " |")
    unknown = [r for r in table if r["precision"] == "unknown"]
    print(f"\n{len(table)} jobs, {len(unknown)} with precision unknown")


if __name__ == "__main__":
    main()
