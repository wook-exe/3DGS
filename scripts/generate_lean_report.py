from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lean-startup"
REPORT_PATH = ROOT / "docs" / "lean-startup-weekly-report.md"
DECISION_PATH = ROOT / "docs" / "adr" / "0001-lean-startup-persevere-decision.md"


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def summarize_feedback(feedback: dict) -> dict:
    participants = feedback["participants"]
    success_rate = sum(1 for item in participants if item["task_success"]) / len(participants)
    satisfaction = average([item["satisfaction"] for item in participants])
    time_to_value = average([item["time_to_value_seconds"] for item in participants])
    compact = [item for item in participants if item["variant"] == "compact"]
    control = [item for item in participants if item["variant"] == "control"]

    return {
        "participants": len(participants),
        "success_rate": round(success_rate, 2),
        "avg_satisfaction": satisfaction,
        "avg_time_to_value_seconds": time_to_value,
        "compact_success_rate": round(
            sum(1 for item in compact if item["task_success"]) / len(compact), 2
        ),
        "control_success_rate": round(
            sum(1 for item in control if item["task_success"]) / len(control), 2
        ),
    }


def summarize_ab_metrics(metrics: dict) -> dict:
    variants = metrics["variants"]
    result = {}

    for variant, payload in variants.items():
        weeks = [payload["week_1"], payload["week_2"]]
        result[variant] = {
            "task_success_rate": average([week["task_success_rate"] for week in weeks]),
            "avg_time_to_value_seconds": average(
                [week["avg_time_to_value_seconds"] for week in weeks]
            ),
            "avg_satisfaction": average([week["avg_satisfaction"] for week in weeks]),
            "event_count": sum(week["event_count"] for week in weeks),
        }

    result["delta"] = {
        "task_success_rate": round(
            result["compact"]["task_success_rate"] - result["control"]["task_success_rate"],
            2,
        ),
        "time_to_value_seconds": round(
            result["compact"]["avg_time_to_value_seconds"]
            - result["control"]["avg_time_to_value_seconds"],
            2,
        ),
        "satisfaction": round(
            result["compact"]["avg_satisfaction"] - result["control"]["avg_satisfaction"],
            2,
        ),
    }
    return result


def decide(summary: dict) -> str:
    delta = summary["delta"]
    if (
        delta["task_success_rate"] >= 0.2
        and delta["time_to_value_seconds"] <= -30
        and delta["satisfaction"] >= 0.5
    ):
        return "PERSEVERE"
    return "PIVOT"


def write_report(feedback: dict, metrics: dict, backlog: dict) -> None:
    feedback_summary = summarize_feedback(feedback)
    ab_summary = summarize_ab_metrics(metrics)
    decision = decide(ab_summary)

    report = f"""# Lean Startup Weekly Experiment Report

Experiment: `{feedback["experiment_id"]}`

## Hypothesis

If the dashboard exposes compact release controls and experiment status, users will understand the system value faster and complete the evaluation scenario more successfully.

## User Feedback

- Participants: {feedback_summary["participants"]}
- Overall task success rate: {feedback_summary["success_rate"]:.0%}
- Compact variant success rate: {feedback_summary["compact_success_rate"]:.0%}
- Control variant success rate: {feedback_summary["control_success_rate"]:.0%}
- Average satisfaction: {feedback_summary["avg_satisfaction"]}
- Average time to value: {feedback_summary["avg_time_to_value_seconds"]} seconds

## A/B Test Result

| Metric | Control | Compact | Delta |
| --- | ---: | ---: | ---: |
| Task success rate | {ab_summary["control"]["task_success_rate"]:.0%} | {ab_summary["compact"]["task_success_rate"]:.0%} | {ab_summary["delta"]["task_success_rate"]:+.0%} |
| Time to value | {ab_summary["control"]["avg_time_to_value_seconds"]}s | {ab_summary["compact"]["avg_time_to_value_seconds"]}s | {ab_summary["delta"]["time_to_value_seconds"]}s |
| Satisfaction | {ab_summary["control"]["avg_satisfaction"]} | {ab_summary["compact"]["avg_satisfaction"]} | {ab_summary["delta"]["satisfaction"]:+.2f} |
| Event count | {ab_summary["control"]["event_count"]} | {ab_summary["compact"]["event_count"]} | {ab_summary["compact"]["event_count"] - ab_summary["control"]["event_count"]:+d} |

## Backlog Status

| ID | Title | Status |
| --- | --- | --- |
"""

    for item in backlog["items"]:
        report += f"| {item['id']} | {item['title']} | {item['status']} |\n"

    report += f"""
## Decision

Decision: **{decision}**

The compact variant improved task success, reduced time to value, and increased satisfaction. Continue the compact release-control direction, while adding a clearer user-facing 360 viewer scenario for non-technical personas.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    DECISION_PATH.write_text(
        f"""# ADR 0001: Lean Startup {decision} Decision

## Status

Accepted

## Context

The 3DGS-Viewer team ran a 2-week feature-flagged A/B experiment using LLM-simulated persona feedback from 10 users. The goal was to validate whether compact release-control messaging helped users understand system readiness faster.

## Decision

We will **{decision}** with the compact release-control direction.

## Evidence

- Compact task success rate exceeded control by {ab_summary["delta"]["task_success_rate"]:+.0%}.
- Compact reduced time to value by {abs(ab_summary["delta"]["time_to_value_seconds"])} seconds.
- Compact satisfaction improved by {ab_summary["delta"]["satisfaction"]:+.2f}.

## Consequences

- Keep the compact dashboard/release-control experience behind `enhanced_dashboard_cards` until production rollout.
- Add a clearer 360 viewer user scenario for non-technical personas before broad release.
- Continue weekly reporting through GitHub Actions.
""",
        encoding="utf-8",
    )


def main() -> None:
    feedback = load_json("persona-feedback.json")
    metrics = load_json("ab-test-metrics.json")
    backlog = load_json("experiment-backlog.json")
    write_report(feedback, metrics, backlog)
    print(REPORT_PATH)
    print(DECISION_PATH)


if __name__ == "__main__":
    main()
