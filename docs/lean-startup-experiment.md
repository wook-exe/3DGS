# Lean Startup Experiment Operation

This document describes the Lean Startup experiment for the 3DGS-Viewer assignment.

## Build-Measure-Learn Loop

Build:

- Add compact release-control messaging behind the `enhanced_dashboard_cards` feature flag.
- Use A/B experiment `dashboard_chart_density` with `control` and `compact` variants.
- Track evaluation events through `/events`.

Measure:

- Collect 10 persona-based LLM-simulated user feedback records.
- Measure task success rate, time to value, satisfaction, and event count.
- Generate weekly reports through GitHub Actions.

Learn:

- Compare the compact variant with the control variant.
- Document Pivot or Persevere decision in ADR form.

## User Scenario

Users review the dashboard, identify release readiness, inspect Feature Flag/A-B status, and decide whether the viewer workflow is ready for small-product 3D scan operation.

## Feedback Dataset

Dataset:

```text
data/lean-startup/persona-feedback.json
```

The dataset contains 10 LLM-simulated users with distinct personas:

- AI engineering student
- MLOps teaching assistant
- Manufacturing QA operator
- Frontend developer
- Backend developer
- Project manager
- Open-source reviewer
- Computer vision researcher
- Cloud operations engineer
- Non-technical exhibition curator

## Two-Week A/B Test

Metric data:

```text
data/lean-startup/ab-test-metrics.json
```

Variants:

- `control`: original DORA dashboard layout
- `compact`: compact release-control and faster status summary

Primary metric:

- `task_success_rate`

Secondary metrics:

- `time_to_value_seconds`
- `satisfaction`
- `event_count`

## Experiment Backlog

Backlog:

```text
data/lean-startup/experiment-backlog.json
```

This mirrors a GitHub Issues/Projects board with labels such as:

- `experiment`
- `user-feedback`
- `metrics`
- `decision`

## Weekly Reporting Automation

Workflow:

```text
.github/workflows/lean-startup-report.yml
```

Script:

```text
scripts/generate_lean_report.py
```

Generated outputs:

```text
docs/lean-startup-weekly-report.md
docs/adr/0001-lean-startup-persevere-decision.md
```

## Decision

The simulated 2-week result supports **PERSEVERE** for compact release-control messaging because it improved task success, reduced time to value, and increased satisfaction.
