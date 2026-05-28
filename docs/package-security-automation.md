# GitHub Packages and Dependency Security Automation

This document records the package, container, and dependency-security automation added for the 3DGS-Viewer assignment.

## npm Package

- Package path: `viewer/`
- Package name: `@wook-exe/3dgs-viewer`
- Current version: `1.0.1`
- Registry: GitHub Packages (`https://npm.pkg.github.com`)

The version update path is documented in `viewer/CHANGELOG.md` as `1.0.0 -> 1.0.1`.

Local verification:

```bash
cd viewer
npm ci
npm test
npm run build
npm run pack:check
```

Publish automation:

- Workflow: `.github/workflows/packages-and-containers.yml`
- Trigger: push to `main`, semantic version tags, or manual dispatch
- Token: `GITHUB_TOKEN` is used as `NODE_AUTH_TOKEN`
- Duplicate publish protection: the workflow checks whether the package version already exists before calling `npm publish`

## Docker Image

- Runtime: `python:3.11-slim`
- Local port: `8000`
- Image target: `ghcr.io/<owner>/3dgs-viewer`

Local verification:

```bash
docker build -t 3dgs-viewer:local .
docker run --rm -p 8000:8000 3dgs-viewer:local
```

Docker Desktop or another Docker daemon must be running before the local verification commands are executed.

Then open:

```text
http://127.0.0.1:8000/1.html
```

CI automation:

- Build local image
- Run smoke test with `curl`
- Push GHCR tags on push events: branch, tag, SHA, and `latest` for the default branch

## Dependabot Policy

Config file: `.github/dependabot.yml`

Enabled ecosystems:

- npm: `/viewer`
- pip: `/`
- Docker: `/`
- GitHub Actions: `/`

Schedule:

- Weekly on Monday morning in `Asia/Seoul`
- Dependency grouping is enabled for npm, pip, and GitHub Actions

Auto-merge policy:

- Workflow: `.github/workflows/dependabot-auto-merge.yml`
- Only Dependabot PRs are eligible
- Only semantic patch updates are auto-merged
- Auto-merge still waits for required branch protection checks

## Security Scanning

Workflow: `.github/workflows/dependency-security.yml`

Scanners:

- `npm audit --audit-level=moderate`
- Snyk via `npx snyk test` when `SNYK_TOKEN` is configured

Outputs:

- Markdown summary in the GitHub Actions job summary
- Artifact: `dependency-security-report`
- GitHub Issue: created or updated when moderate-or-higher vulnerabilities are detected on non-PR runs

Required repository secret:

- `SNYK_TOKEN`: optional for local coursework demos, required for full Snyk scanning in CI
