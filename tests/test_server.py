import json
from io import BytesIO

import src.server as server_module
from src.server import ViewerRequestHandler


class FakeRequestHandler(ViewerRequestHandler):
    def __init__(self, path="/health", body=b""):
        self.path = path
        self.request_version = "HTTP/1.1"
        self.command = "GET"
        self.wfile = BytesIO()
        self.rfile = BytesIO(body)
        self.response_code = None
        self.headers_sent = {}
        self.headers = {"Content-Length": str(len(body))}

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        self.headers_sent[keyword] = value

    def end_headers(self):
        pass


def test_health_endpoint_returns_ok_payload():
    handler = FakeRequestHandler("/health")

    handler.do_GET()

    assert handler.response_code == 200
    assert handler.headers_sent["Content-Type"] == "application/json"
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload == {"status": "ok", "service": "3dgs-viewer"}


def test_flags_endpoint_returns_feature_and_experiment_assignments():
    handler = FakeRequestHandler("/flags?user_id=student-001")

    handler.do_GET()

    assert handler.response_code == 200
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["user_id"] == "student-001"
    assert "model_status_sidebar" in payload["features"]
    assert "dashboard_chart_density" in payload["experiments"]


def test_rollout_endpoint_returns_canary_stages():
    handler = FakeRequestHandler("/rollout")

    handler.do_GET()

    assert handler.response_code == 200
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["strategy"] == "canary"
    assert payload["stages"] == [1, 10, 50, 100]


def test_events_endpoint_tracks_experiment_event(monkeypatch):
    def fake_track_event(**kwargs):
        return {
            "timestamp": "2026-05-28T00:00:00+00:00",
            **kwargs,
        }

    monkeypatch.setattr(server_module, "track_event", fake_track_event)
    body = json.dumps(
        {
            "event_name": "dashboard_loaded",
            "user_id": "student-001",
            "experiment_key": "dashboard_chart_density",
            "properties": {"source": "unit"},
        }
    ).encode("utf-8")
    handler = FakeRequestHandler("/events", body)

    handler.do_POST()

    assert handler.response_code == 201
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["ok"] is True
    assert payload["event"]["variant"] in {"control", "compact"}


def test_events_endpoint_rejects_invalid_payload():
    handler = FakeRequestHandler("/events", b"[]")

    handler.do_POST()

    assert handler.response_code == 400
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["error"] == "invalid_event"


def test_post_to_unknown_endpoint_returns_not_found():
    handler = FakeRequestHandler("/unknown", b"{}")

    handler.do_POST()

    assert handler.response_code == 404
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["error"] == "not_found"
