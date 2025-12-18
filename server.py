import json
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class SplendorHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        # Quieter logging for a cleaner console
        return

    def do_POST(self):
        if self.path != "/py/decide_action":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}

        difficulty = payload.get("difficulty", payload.get("aiLevel", 2))
        try:
            difficulty_int = int(difficulty)
        except (TypeError, ValueError):
            difficulty_int = 2

        action = {"type": "endTurn", "difficulty": difficulty_int}

        body = json.dumps({"action": action}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(port: int = 8000):
    with socketserver.TCPServer(("", port), SplendorHandler) as httpd:
        print(f"Serving on http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run()
