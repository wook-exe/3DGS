import pytest

from src.quality_metrics import (
    average_processing_time,
    calculate_pass_rate,
    classify_inspection,
    count_failures,
    recent_inspections,
    summarize_by_gasket_type,
)


SAMPLE_RECORDS = [
    {
        "timestamp": "2026-05-28T09:00:00",
        "gasket_type": "OPEN",
        "verdict": "PASS",
        "score": 41.2,
        "processing_time_ms": 118,
    },
    {
        "timestamp": "2026-05-28T09:01:00",
        "gasket_type": "OPEN",
        "verdict": "FAIL",
        "score": 48.7,
        "processing_time_ms": 131,
    },
    {
        "timestamp": "2026-05-28T09:02:00",
        "gasket_type": "CLOSE",
        "verdict": "PASS",
        "score": 0.01,
        "processing_time_ms": 96,
    },
    {
        "timestamp": "2026-05-28T09:03:00",
        "gasket_type": "CLOSE",
        "verdict": "FAIL",
        "score": 0.09,
        "processing_time_ms": 143,
    },
]


def test_classify_inspection_uses_threshold_boundary():
    assert classify_inspection(46.44, 46.45) == "PASS"
    assert classify_inspection(46.45, 46.45) == "FAIL"


def test_classify_inspection_rejects_negative_inputs():
    with pytest.raises(ValueError):
        classify_inspection(-1, 46.45)
    with pytest.raises(ValueError):
        classify_inspection(1, -0.1)


def test_calculate_pass_rate_returns_percentage():
    assert calculate_pass_rate(SAMPLE_RECORDS) == 50.0
    assert calculate_pass_rate([]) == 0.0


def test_count_failures_counts_fail_verdicts():
    assert count_failures(SAMPLE_RECORDS) == 2


def test_average_processing_time_rounds_to_two_decimals():
    assert average_processing_time(SAMPLE_RECORDS) == 122.0
    assert average_processing_time([]) == 0.0


def test_summarize_by_gasket_type_counts_pass_and_fail():
    assert summarize_by_gasket_type(SAMPLE_RECORDS) == {
        "OPEN": {"PASS": 1, "FAIL": 1},
        "CLOSE": {"PASS": 1, "FAIL": 1},
    }


def test_recent_inspections_sorts_descending_and_limits_results():
    recent = recent_inspections(SAMPLE_RECORDS, limit=2)

    assert [item["timestamp"] for item in recent] == [
        "2026-05-28T09:03:00",
        "2026-05-28T09:02:00",
    ]


def test_recent_inspections_rejects_invalid_limit():
    with pytest.raises(ValueError):
        recent_inspections(SAMPLE_RECORDS, limit=0)
