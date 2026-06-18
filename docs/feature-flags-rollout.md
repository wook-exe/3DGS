# Feature Flags, A/B Testing, and Canary Rollout

This assignment step adds runtime release controls to the 3DGS-Viewer app.

## Feature Flags

Implementation: `src/feature_flags.py`

Flags:

- `enhanced_dashboard_cards`
- `model_status_sidebar`
- `experiment_event_tracking`
- `canary_release_panel`

Each flag supports:

- Global environment override: `FF_<FLAG_KEY>=true|false`
- Target user override: `FF_<FLAG_KEY>_USERS=user-a,user-b`
- Environment targeting: `FF_<FLAG_KEY>_ENVS=staging,production`

Example:

```powershell
$env:APP_ENV="staging"
$env:FF_ENHANCED_DASHBOARD_CARDS_USERS="student-001,qa-user"
$env:FF_CANARY_RELEASE_PANEL_ENVS="staging"
python -m src.server
```

Runtime API:

```text
GET /flags?user_id=student-001
```

## A/B Testing

Implementation: `src/ab_testing.py`

Experiments:

- `dashboard_chart_density`: `control` vs `compact`
- `model_status_copy`: `baseline` vs `risk_focused`

Assignment method:

- Stable SHA-256 hash of `experiment_key:user_id`
- 50/50 weighted bucket split
- Same user always receives the same variant for the same experiment

Event tracking endpoint:

```text
POST /events
```

Example payload:

```json
{
  "event_name": "dashboard_loaded",
  "user_id": "student-001",
  "experiment_key": "dashboard_chart_density",
  "properties": {
    "source": "github-pages-demo"
  }
}
```

Events are appended to:

```text
data/experiment-events.jsonl
```

## Canary Rollout

Rollout config: `config/canary-rollout.yml`

Stages:

- 1%
- 10%
- 50%
- 100%

Rollback trigger:

- `/health` success rate below `99%`
- p95 latency above `500 ms`

Simulation:

```powershell
python scripts/simulate_canary_rollout.py
```

Output:

```text
data/canary-rollout-simulation.json
```

The scenario validates progressive promotion while preserving an automatic rollback rule for unhealthy stages.
