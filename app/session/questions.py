"""Question bank loader: questions/bank.json (status: DRAFT, needs author review)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

BANK = Path(__file__).resolve().parents[2] / "questions" / "bank.json"
FIELDS = {"id", "category", "text", "type", "suggested_time_s"}
TYPES = {"behavioral", "technical"}


def load_bank(path: Path = BANK) -> dict:
    """The bank as a dict, checked: every question has the fields, a known category and type."""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = set()
    for q in bank["questions"]:
        missing = FIELDS - set(q)
        if missing:
            raise ValueError(f"question {q.get('id')} is missing {sorted(missing)}")
        if q["category"] not in bank["categories"]:
            raise ValueError(f"question {q['id']} has unknown category {q['category']}")
        if q["type"] not in TYPES:
            raise ValueError(f"question {q['id']} has unknown type {q['type']}")
        if not (isinstance(q["suggested_time_s"], (int, float)) and q["suggested_time_s"] > 0):
            raise ValueError(f"question {q['id']} needs a positive suggested_time_s")
        if q["id"] in ids:
            raise ValueError(f"duplicate question id {q['id']}")
        ids.add(q["id"])
    return bank


@lru_cache(maxsize=1)
def default_bank() -> dict:
    return load_bank()


def find_question(bank: dict, question_id: str) -> dict:
    for q in bank["questions"]:
        if q["id"] == question_id:
            return q
    raise KeyError(f"no question {question_id}")
