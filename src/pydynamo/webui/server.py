"""Local HTTP server between the HTML view and the dcp session (pydynamo.webui.api.Session).

Binds 127.0.0.1 on a free port. Every request must carry the session's random token (cookie `pydynamo_t`, set when the
window first opens /?t=<token>) and a Host header naming this server, so other users of the node cannot drive the
session. The token never appears on a command line.

  GET  /?t=TOKEN               sets the cookie, redirects to /
  GET  /                       the view (static/index.html); /static/...  its files
  GET  /api/meta               constants of the dcp windows
  GET  /api/state              project, step buttons, window fields
  GET  /api/messages?after=N   messages of the main window after sequence number N
  GET  /api/slices?which=mask&axis=z&k=33[&mfile=F]   template and mask tiles (mask inspector)
  GET  /api/gpus               the GPUs of this machine (nvidia-smi)
  GET  /api/slurm?gpus=N        where a Slurm job with N GPUs can run (sinfo, read only)
  GET  /api/fs?path=DIR        folder listing for the file chooser
  POST /api/call               {"method": ..., "args": [...], "kwargs": {...}}  -> {"ok", "result" | "error"}
"""
import hmac
import json
import mimetypes
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pydynamo.webui.api import ApiError, Session

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2", ".woff": "font/woff",
         ".png": "image/png", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "pydynamo-webui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):               # quiet; errors reach the view as messages
        if os.environ.get("PYDYNAMO_WEBUI_DEBUG"):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------------------------------------ plumbing
    def _send(self, code, body=b"", ctype="application/json", headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _host_ok(self):
        host = (self.headers.get("Host") or "").strip()
        return host in self.server.hosts

    def _token_ok(self):
        ck = self.headers.get("Cookie") or ""
        for part in ck.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "pydynamo_t" and hmac.compare_digest(v, self.server.token):
                return True
        return False

    def _authorised(self):
        if not self._host_ok() or not self._token_ok():
            self._send(403, "forbidden", "text/plain")
            return False
        return True

    # ------------------------------------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" and "t" in q:                # first load: token -> cookie
            if not self._host_ok() or not hmac.compare_digest(q["t"][0], self.server.token):
                return self._send(403, "forbidden", "text/plain")
            return self._send(303, "", "text/plain", {
                "Location": "/", "Set-Cookie": "pydynamo_t=%s; Path=/; HttpOnly; SameSite=Strict" % self.server.token})
        if not self._authorised():
            return
        s = self.server.session
        try:
            if u.path in ("/", "/index.html"):
                return self._static("index.html")
            if u.path.startswith("/static/"):
                return self._static(u.path[len("/static/"):])
            if u.path == "/api/meta":
                return self._json(s.meta())
            if u.path == "/api/state":
                return self._json(s.state())
            if u.path == "/api/messages":
                return self._json(s.messages(int(q.get("after", ["0"])[0])))
            if u.path == "/api/slices":
                with s.lock:
                    return self._json(s.slices(q.get("which", ["mask"])[0], q.get("axis", ["z"])[0],
                                               q.get("k", [None])[0], q.get("mfile", [None])[0]))
            if u.path == "/api/gpus":
                return self._json(s.gpus())
            if u.path == "/api/slurm":
                return self._json(s.slurm_plan(q.get("gpus", ["1"])[0]))
            if u.path == "/api/fs":
                return self._json(s.listdir(q.get("path", ["."])[0]))
        except ApiError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:                       # pragma: no cover - reported to the view
            return self._json({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}, 500)
        self._send(404, "not found", "text/plain")

    def _static(self, rel):
        path = os.path.normpath(os.path.join(STATIC, rel))
        if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
            return self._send(404, "not found", "text/plain")
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            body = f.read()
        self._send(200, body, TYPES.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream")

    # ------------------------------------------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        if not self._authorised():
            return
        if u.path != "/api/call":
            return self._send(404, "not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            out = self.server.session.call(req["method"], req.get("args") or [], req.get("kwargs") or {})
        except ApiError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        except (KeyError, ValueError) as e:
            return self._json({"ok": False, "error": "bad request: %s" % e}, 400)
        self._json(out)


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session, port=0):
        super().__init__(("127.0.0.1", port), _Handler)
        self.session = session
        self.token = secrets.token_urlsafe(24)
        p = self.server_address[1]
        self.hosts = {"127.0.0.1:%d" % p, "localhost:%d" % p}

    @property
    def url(self):
        return "http://127.0.0.1:%d/" % self.server_address[1]

    @property
    def start_url(self):
        return "%s?t=%s" % (self.url, self.token)

    def start(self):
        th = threading.Thread(target=self.serve_forever, name="pydynamo-webui", daemon=True)
        th.start()
        return th


def serve(project=None, port=0):
    """A started server with a fresh session (for the Qt window, tests and `dcp --serve`)."""
    srv = WebServer(Session(project), port)
    srv.start()
    return srv
