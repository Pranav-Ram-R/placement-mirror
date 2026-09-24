"""Measurement schema.

Every performance number in this project is a Measurement with a Source.
Numbers computed from other numbers are Derived and carry their formula.
"""

from __future__ import annotations

import json
import numbers
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Source(Enum):
    """Where a measurement was taken."""

    AIHUB_X_ELITE = "AI Hub hosted Snapdragon X Elite"
    LOCAL_X86_CPU = "local x86 CPU"
    PHYSICAL_SNAPDRAGON = "physical Snapdragon device"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: object, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{owner}.value must be a real number, got {type(value).__name__}")
    return float(value)


PRECISIONS = frozenset({"float32", "float16", "w8a8", "w8a16", "w4a16", "int8", "unknown"})


@dataclass(frozen=True, kw_only=True)
class Measurement:
    """One number from one real run."""

    model: str
    metric: str
    value: float
    unit: str
    source: Source
    runtime: str
    compute_unit: str
    precision: str
    qnn_version: str | None = None
    ort_version: str | None = None
    job_id: str | None = None
    timestamp: str = field(default_factory=_utc_now)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            raise TypeError(f"Measurement.source must be a Source, got {self.source!r}")
        if self.source is Source.AIHUB_X_ELITE and not (self.job_id and self.job_id.strip()):
            raise ValueError("Measurement from AI Hub needs a job_id")
        if not (isinstance(self.runtime, str) and self.runtime.strip()):
            raise ValueError("Measurement.runtime must be a non-empty string")
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"Measurement.precision must be one of {sorted(PRECISIONS)}, got {self.precision!r}"
            )
        object.__setattr__(self, "value", _as_float(self.value, "Measurement"))


@dataclass(frozen=True, kw_only=True)
class Derived:
    """A number computed from measurements. Its formula is always shown."""

    name: str
    value: float
    unit: str
    formula: str
    inputs: list[Measurement]

    def __post_init__(self) -> None:
        if not (isinstance(self.formula, str) and self.formula.strip()):
            raise ValueError("Derived.formula must be a non-empty string")
        if not self.inputs:
            raise ValueError("Derived.inputs needs at least one Measurement")
        if not all(isinstance(m, Measurement) for m in self.inputs):
            raise TypeError("Derived.inputs must contain only Measurement objects")
        object.__setattr__(self, "value", _as_float(self.value, "Derived"))
        object.__setattr__(self, "inputs", list(self.inputs))

    @property
    def label(self) -> str:
        return f"Derived: {self.name} = {self.value:g} {self.unit} (formula: {self.formula})"


def _measurement_to_dict(m: Measurement) -> dict:
    d = asdict(m)
    d["source"] = m.source.value
    return d


def _upgrade_legacy(d: dict) -> dict:
    """Records written before `runtime` existed. Missing facts become "unknown", never a guess."""
    d["runtime"] = "unknown"
    if d.get("precision") not in PRECISIONS:
        legacy = f"legacy precision: {d.get('precision')!r}"
        d["notes"] = f"{d['notes']} | {legacy}" if d.get("notes") else legacy
        d["precision"] = "unknown"
    return d


def _measurement_from_dict(d: dict) -> Measurement:
    d = dict(d)
    if "runtime" not in d:
        d = _upgrade_legacy(d)
    d["source"] = Source(d["source"])
    return Measurement(**d)


def to_dict(record: Measurement | Derived) -> dict:
    if isinstance(record, Measurement):
        return {"kind": "measurement", **_measurement_to_dict(record)}
    if isinstance(record, Derived):
        return {
            "kind": "derived",
            "name": record.name,
            "value": record.value,
            "unit": record.unit,
            "formula": record.formula,
            "inputs": [_measurement_to_dict(m) for m in record.inputs],
        }
    raise TypeError(f"Cannot serialize {type(record).__name__}")


def from_dict(d: dict) -> Measurement | Derived:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind == "measurement":
        return _measurement_from_dict(d)
    if kind == "derived":
        d["inputs"] = [_measurement_from_dict(m) for m in d["inputs"]]
        return Derived(**d)
    raise ValueError(f"Unknown record kind: {kind!r}")


def save_json(records: list[Measurement | Derived], path: str | Path) -> None:
    """Write measurements and derived values to a JSON file."""
    data = [to_dict(r) for r in records]
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> list[Measurement | Derived]:
    """Read measurements and derived values from a JSON file. Validation runs on load."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [from_dict(d) for d in data]
