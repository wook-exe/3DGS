from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "canary-rollout-simulation.json"


@dataclass(frozen=True)
class CanaryStage:
    name: str
    traffic_percent: int
    success_rate: float
    p95_latency_ms: int


STAGES = (
    CanaryStage("canary-1", 1, 1.0, 180),
    CanaryStage("canary-10", 10, 0.998, 210),
    CanaryStage("canary-50", 50, 0.997, 260),
    CanaryStage("full-rollout", 100, 0.996, 290),
)


def evaluate_stage(stage: CanaryStage) -> dict[str, str | int | float | bool]:
    healthy = stage.success_rate >= 0.99 and stage.p95_latency_ms <= 500
    return {
        "stage": stage.name,
        "traffic_percent": stage.traffic_percent,
        "success_rate": stage.success_rate,
        "p95_latency_ms": stage.p95_latency_ms,
        "healthy": healthy,
        "action": "promote" if healthy else "rollback",
    }


def simulate() -> list[dict[str, str | int | float | bool]]:
    results = []
    for stage in STAGES:
        result = evaluate_stage(stage)
        results.append(result)
        if not result["healthy"]:
            break
    return results


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = simulate()
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(OUTPUT)
