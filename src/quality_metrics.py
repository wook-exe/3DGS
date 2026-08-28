from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, TypedDict


Verdict = Literal["PASS", "FAIL"]
GasketType = Literal["OPEN", "CLOSE"]


class InspectionRecord(TypedDict):
    timestamp: str
    gasket_type: GasketType
    verdict: Verdict
    score: float
    processing_time_ms: int


def classify_inspection(score: float, threshold: float) -> Verdict:
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if score < 0:
        raise ValueError("score must be non-negative.")
    return "FAIL" if score >= threshold else "PASS"


def calculate_pass_rate(records: Iterable[InspectionRecord]) -> float:
    items = list(records)
    if not items:
        return 0.0

    pass_count = sum(1 for item in items if item["verdict"] == "PASS")
    return round((pass_count / len(items)) * 100, 2)


def count_failures(records: Iterable[InspectionRecord]) -> int:
    return sum(1 for item in records if item["verdict"] == "FAIL")


def average_processing_time(records: Iterable[InspectionRecord]) -> float:
    items = list(records)
    if not items:
        return 0.0

    total = sum(item["processing_time_ms"] for item in items)
    return round(total / len(items), 2)


def summarize_by_gasket_type(records: Iterable[InspectionRecord]) -> dict[GasketType, dict[str, int]]:
    summary: dict[GasketType, Counter[str]] = {
        "OPEN": Counter({"PASS": 0, "FAIL": 0}),
        "CLOSE": Counter({"PASS": 0, "FAIL": 0}),
    }

    for item in records:
        summary[item["gasket_type"]][item["verdict"]] += 1

    return {
        gasket_type: {"PASS": counts["PASS"], "FAIL": counts["FAIL"]}
        for gasket_type, counts in summary.items()
    }


def recent_inspections(
    records: Iterable[InspectionRecord],
    *,
    limit: int = 20,
) -> list[InspectionRecord]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    return sorted(records, key=lambda item: item["timestamp"], reverse=True)[:limit]
