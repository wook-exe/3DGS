# Shift-Left Testing Automation

This assignment step adds early automated testing gates for the 3DGS-Viewer project.

## Unit Coverage Gate

Workflow:

```text
.github/workflows/shift-left-testing.yml
```

Command:

```bash
pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

The CI build fails when Python source coverage drops below 80%.

## TDD Cycle Evidence

Core module:

```text
src/quality_metrics.py
```

Tests:

```text
tests/test_quality_metrics.py
```

Implemented through Red-Green-Refactor style test cases:

- `classify_inspection`
- `calculate_pass_rate`
- `count_failures`
- `average_processing_time`
- `summarize_by_gasket_type`
- `recent_inspections`

These functions support the operational dashboard by turning inspection rows into KPI and table-ready values.

## Playwright E2E Scenario

Scenario:

```text
e2e/dashboard.spec.js
```

The E2E test starts the local Python server and verifies:

- DORA dashboard heading renders.
- Four Chart.js canvases are present.
- Feature flag release-control panel is visible.
- `/flags` returns stable experiment assignments.
- `/events` accepts an experiment tracking event.

Artifacts:

- `test-results/`
- `playwright-report/`

The workflow uploads Playwright artifacts with `if: always()`, so failure screenshots and traces remain available for review.

## Legacy Refactoring Safety

Legacy utility behavior remains covered by:

```text
tests/test_utils.py
tests/test_server.py
tests/test_feature_flags.py
tests/test_ab_testing.py
```

The coverage gate protects future refactors of the server, feature flags, A/B testing, and dashboard metric calculations.
