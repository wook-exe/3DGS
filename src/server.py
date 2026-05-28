from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    """Serve the static viewer and expose a lightweight health endpoint."""

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "service": "3dgs-viewer",
                }
            )
            return

        super().do_GET()

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
