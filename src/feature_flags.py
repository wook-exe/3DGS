from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class FeatureFlag:
    key: str
    default: bool
    description: str


DEFAULT_FLAGS: dict[str, FeatureFlag] = {
    "enhanced_dashboard_cards": FeatureFlag(
        key="enhanced_dashboard_cards",
        default=False,
        description="Show richer KPI cards on the DORA dashboard.",
    ),
    "model_status_sidebar": FeatureFlag(
        key="model_status_sidebar",
        default=True,
        description="Show OPEN/CLOSE model status details beside the viewer.",
    ),
    "experiment_event_tracking": FeatureFlag(
        key="experiment_event_tracking",
        default=True,
        description="Persist A/B assignment and interaction events.",
    ),
    "canary_release_panel": FeatureFlag(
        key="canary_release_panel",
        default=False,
        description="Expose canary rollout progress in deployment metadata.",
    ),
}


def _env_key(flag_key: str, suffix: str = "") -> str:
    base = f"FF_{flag_key.upper()}"
    return f"{base}_{suffix}" if suffix else base


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def is_feature_enabled(
    flag_key: str,
    *,
    user_id: str | None = None,
    app_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    env = env or os.environ
    app_env = app_env or env.get("APP_ENV", "local")
    flag = DEFAULT_FLAGS[flag_key]

    explicit = _parse_bool(env.get(_env_key(flag_key)))
    if explicit is not None:
        return explicit

    target_users = _split_csv(env.get(_env_key(flag_key, "USERS")))
    if user_id and user_id in target_users:
        return True

    enabled_envs = _split_csv(env.get(_env_key(flag_key, "ENVS")))
    if app_env in enabled_envs:
        return True

    return flag.default


def evaluate_features(
    *,
    user_id: str | None = None,
    app_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    return {
        key: is_feature_enabled(key, user_id=user_id, app_env=app_env, env=env)
        for key in DEFAULT_FLAGS
    }


def feature_metadata() -> list[dict[str, str | bool]]:
    return [
        {
            "key": flag.key,
            "default": flag.default,
            "description": flag.description,
            "env_var": _env_key(flag.key),
            "target_users_var": _env_key(flag.key, "USERS"),
        }
        for flag in DEFAULT_FLAGS.values()
    ]
