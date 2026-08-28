# ADR 0001: Lean Startup PERSEVERE Decision

## Status

Accepted

## Context

The 3DGS-Viewer team ran a 2-week feature-flagged A/B experiment using LLM-simulated persona feedback from 10 users. The goal was to validate whether compact release-control messaging helped users understand system readiness faster.

## Decision

We will **PERSEVERE** with the compact release-control direction.

## Evidence

- Compact task success rate exceeded control by +40%.
- Compact reduced time to value by 68.1 seconds.
- Compact satisfaction improved by +1.40.

## Consequences

- Keep the compact dashboard/release-control experience behind `enhanced_dashboard_cards` until production rollout.
- Add a clearer 360 viewer user scenario for non-technical personas before broad release.
- Continue weekly reporting through GitHub Actions.
