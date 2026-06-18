from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVENT_LOG_PATH = ROOT / "data" / "experiment-events.jsonl"


@dataclass(frozen=True)
class Variant:
    key: str
    weight: int


@dataclass(frozen=True)
class Experiment:
    key: str
    variants: tuple[Variant, ...]


EXPERIMENTS: dict[str, Experiment] = {
    "dashboard_chart_density": Experiment(
        key="dashboard_chart_density",
        variants=(
            Variant("control", 50),
            Variant("compact", 50),
        ),
    ),
    "model_status_copy": Experiment(
        key="model_status_copy",
        variants=(
            Variant("baseline", 50),
            Variant("risk_focused", 50),
        ),
    ),
}


def _bucket(experiment_key: str, user_id: str) -> int:
    digest = hashlib.sha256(f"{experiment_key}:{user_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def assign_variant(experiment_key: str, user_id: str) -> str:
    experiment = EXPERIMENTS[experiment_key]
    bucket = _bucket(experiment_key, user_id)
    cumulative = 0

    for variant in experiment.variants:
        cumulative += variant.weight
        if bucket < cumulative:
            return variant.key

    return experiment.variants[-1].key


def assignments_for_user(user_id: str) -> dict[str, str]:
    return {
        experiment_key: assign_variant(experiment_key, user_id)
        for experiment_key in EXPERIMENTS
    }


def track_event(
    *,
    event_name: str,
    user_id: str,
    experiment_key: str,
    variant: str,
    properties: dict[str, Any] | None = None,
    log_path: Path = EVENT_LOG_PATH,
) -> dict[str, Any]:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_name": event_name,
        "user_id": user_id,
        "experiment_key": experiment_key,
        "variant": variant,
        "properties": properties or {},
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")

    return event
