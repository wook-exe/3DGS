import json
from io import BytesIO

from src.server import ViewerRequestHandler


class HealthCheckHandler(ViewerRequestHandler):
    def __init__(self):
        self.path = "/health"
        self.request_version = "HTTP/1.1"
        self.command = "GET"
        self.wfile = BytesIO()
        self.response_code = None
        self.headers_sent = {}

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        self.headers_sent[keyword] = value

    def end_headers(self):
        pass


def test_health_endpoint_returns_ok_payload():
    handler = HealthCheckHandler()

    handler.do_GET()

    assert handler.response_code == 200
    assert handler.headers_sent["Content-Type"] == "application/json"
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload == {"status": "ok", "service": "3dgs-viewer"}
