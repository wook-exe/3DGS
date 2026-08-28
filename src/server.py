from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.ab_testing import assign_variant, assignments_for_user, track_event
from src.feature_flags import evaluate_features, feature_metadata


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    """Serve the static viewer and expose a lightweight health endpoint."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "3dgs-viewer",
                }
            )
            return

        if path == "/flags":
            user_id = query.get("user_id", ["anonymous"])[0]
            self._send_json(
                {
                    "user_id": user_id,
                    "features": evaluate_features(user_id=user_id),
                    "feature_metadata": feature_metadata(),
                    "experiments": assignments_for_user(user_id),
                }
            )
            return

        if path == "/rollout":
            self._send_json(
                {
                    "service": "3dgs-viewer",
                    "strategy": "canary",
                    "stages": [1, 10, 50, 100],
                    "health_check": {
                        "path": "/health",
                        "success_rate_threshold": 0.99,
                        "max_p95_latency_ms": 500,
                    },
                    "rollback": {
                        "enabled": True,
                        "trigger": "health_check_failed",
                    },
                }
            )
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/events":
            self._send_json({"error": "not_found"}, status_code=404)
            return

        try:
            payload = self._read_json_body()
            user_id = str(payload.get("user_id", "anonymous"))
            experiment_key = str(payload["experiment_key"])
            variant = str(payload.get("variant") or assign_variant(experiment_key, user_id))
            event = track_event(
                event_name=str(payload.get("event_name", "experiment_event")),
                user_id=user_id,
                experiment_key=experiment_key,
                variant=variant,
                properties=payload.get("properties") or {},
            )
        except (KeyError, ValueError, json.JSONDecodeError, TypeError) as exc:
            self._send_json({"error": "invalid_event", "detail": str(exc)}, status_code=400)
            return

        self._send_json({"ok": True, "event": event}, status_code=201)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    app_root = Path(__file__).resolve().parents[1]
    handler = lambda *args, **kwargs: ViewerRequestHandler(  # noqa: E731
        *args,
        directory=str(app_root),
        **kwargs,
    )

    with ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"Serving 3DGS Viewer on http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
