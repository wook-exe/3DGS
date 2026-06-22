import json
from io import BytesIO

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
