# Multiplatform Deployment Automation

This document describes the frontend, container, and cloud deployment automation for the 3DGS-Viewer assignment.

## Frontend Deployment

Workflow: `.github/workflows/frontend-pages.yml`

Platform: GitHub Pages through the `gh-pages` branch.

Generated output:

- `1.html` is copied to `public/index.html`.
- `.nojekyll` is added for plain static hosting.
- `deployment-metadata.json` is generated for deployment traceability.

Production URL:

```text
https://wook-exe.github.io/3DGS/
```

PR preview URL pattern:

```text
https://wook-exe.github.io/3DGS/pr-<PR_NUMBER>/
```

The workflow comments the preview URL on the pull request and removes the preview directory when the PR is closed.

Required GitHub repository setting:

```text
Settings > Pages > Build and deployment > Source: Deploy from a branch
Branch: gh-pages / root
```

## Docker Deployment Strategy

The repository uses a single immutable container image as the deployable unit.

Pipeline stages:

1. Build image from `Dockerfile`.
2. Smoke-test the container locally in CI.
3. Push image to GitHub Container Registry.
4. Deploy the same container digest to the external cloud environment.
5. Verify `/health` after deployment.

Image tags:

- `latest` for the default branch.
- branch name for branch builds.
- semantic version tags for releases.
- `sha-<commit>` for immutable rollbacks.

Runtime health endpoint:

```text
/health
```

Expected response:

```json
{"status": "ok", "service": "3dgs-viewer"}
```

## External Cloud Deployment

Workflow: `.github/workflows/cloud-run-deploy.yml`

Platform: Google Cloud Run.

Deployment behavior:

- Build the Docker image.
- Push it to Google Artifact Registry.
- Deploy it to Cloud Run.
- Run a live health check against `<SERVICE_URL>/health`.
- Write the service URL and image URI to the GitHub Actions job summary.

Required repository variable:

```text
CLOUD_RUN_ENABLED=true
GCP_PROJECT_ID=<your-gcp-project-id>
GCP_REGION=asia-northeast3
GCP_ARTIFACT_REPOSITORY=3dgs-viewer
CLOUD_RUN_SERVICE=3dgs-viewer
```

Required repository secret:

```text
GCP_SA_KEY=<Google service account JSON key>
```

Minimum service account roles:

```text
Artifact Registry Admin
Cloud Run Admin
Service Account User
```

Monitoring:

- Cloud Run request count, latency, error rate, and container instance metrics are available in Cloud Monitoring.
- Application logs are available in Cloud Logging.
- The Actions workflow performs deployment-time health verification with `/health`.

Optional uptime check:

```bash
gcloud monitoring uptime create 3dgs-viewer-health \
  --resource-type=uptime-url \
  --resource-labels=host=<cloud-run-host>,project_id=<project-id> \
  --path=/health
```
