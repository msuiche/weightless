#!/usr/bin/env python3
"""dspark-router — single OpenAI-compatible front door for the 2x DGX Spark rig.

Routes /v1/* requests to the per-lane vLLM servers by the request's "model"
field, and aggregates /v1/models so clients (hermes, omp, scripts) see every
local lane through one base_url: http://spark-4687.local:8000/v1

Stdlib-only (the head has no aiohttp/httpx). Threading HTTP server, SSE
streaming passthrough.
"""
import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = ("0.0.0.0", 8000)
UPSTREAM_HOST = "127.0.0.1"

# lane model id -> upstream port (see weightless/recipe/*/.env.*)
ROUTES = {
    "qwen38-nvfp4": 8078,                # Qwen3.8-27B NVFP4, 1x Spark
    "qwen38-flash-next-nvfp4": 8079,     # Qwen3.8-Flash-Next NVFP4, TP=2
    "glm53-flash": 8080,                 # GLM-5.3-Flash NVFP4, TP=2
    "glm-5.3": 8081,                     # GLM-5.3 743B, TP=4
    "inkling-small-nvfp4": 8082,         # Inkling-Small NVFP4, TP=2
    "deepseek-v4-flash-dspark": 8888,    # DSV4 Flash 0731 NVFP4, TP=2
}

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te",
              "trailer", "upgrade", "proxy-authorization", "proxy-authenticate"}


def upstream_models():
    """Merge /v1/models from every lane that answers within 1s."""
    out = []
    for port in sorted(set(ROUTES.values())):
        try:
            c = http.client.HTTPConnection(UPSTREAM_HOST, port, timeout=1)
            c.request("GET", "/v1/models")
            r = c.getresponse()
            if r.status == 200:
                out.extend(json.loads(r.read()).get("data", []))
            c.close()
        except Exception:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models",):
            data = upstream_models()
            if not data:  # nothing live: still advertise the lanes
                data = [{"id": m, "object": "model", "owned_by": "weightless"}
                        for m in sorted(ROUTES)]
            self._send_json(200, {"object": "list", "data": data})
        elif self.path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "lanes": sorted(ROUTES)})
        else:
            self._send_json(404, {"error": "unknown path"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        model = None
        try:
            model = json.loads(body or b"{}").get("model")
        except Exception:
            pass
        port = ROUTES.get(model)
        if port is None:
            self._send_json(400, {
                "error": f"unknown or missing model {model!r}; "
                         f"known lanes: {sorted(ROUTES)}"})
            return
        try:
            c = http.client.HTTPConnection(UPSTREAM_HOST, port, timeout=None)
            hdrs = {k: v for k, v in self.headers.items()
                    if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
            hdrs["Content-Length"] = str(len(body))
            c.request("POST", self.path, body=body, headers=hdrs)
            r = c.getresponse()
        except Exception as e:
            self._send_json(502, {"error": f"lane :{port} unreachable: {e}"})
            return
        self.send_response(r.status)
        is_sse = "text/event-stream" in (r.getheader("Content-Type") or "")
        for k, v in r.getheaders():
            if k.lower() not in HOP_BY_HOP and k.lower() != "content-length":
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if is_sse:
                self._relay_sse(r)
            else:
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            c.close()

    def _relay_sse(self, r):
        """Stream SSE events, stripping inkling's message wrapper tokens.

        vLLM 0.28.0's --reasoning-parser inkling leaks <|message_model|>,
        <|content_text|> and <|end_message|> into delta.content when streaming
        (the non-streaming path is clean). Strip them here; markers can split
        across events, so content goes through a stateful stripper.
        """
        strip = MarkerStripper()
        buf = b""

        def emit(b):
            self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
            self.wfile.flush()

        while True:
            part = r.read1(65536)
            if not part:
                break
            buf += part
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                emit(self._filter_sse_line(line, strip) + b"\n")
        if buf:
            emit(self._filter_sse_line(buf, strip) + b"\n")

    @staticmethod
    def _filter_sse_line(line, strip):
        if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
            return line
        try:
            ev = json.loads(line[5:])
        except Exception:
            return line
        dirty = False
        for ch in ev.get("choices", []):
            delta = ch.get("delta") or {}
            c = delta.get("content")
            if c:
                nc = strip.feed(c)
                if nc != c:
                    delta["content"] = nc
                    dirty = True
        return b"data: " + json.dumps(ev).encode() if dirty else line


class MarkerStripper:
    """Stateful filter dropping fixed marker strings that may straddle feeds."""

    MARKERS = ("<|message_model|>", "<|content_text|>", "<|end_message|>")
    MAXLEN = max(len(m) for m in MARKERS)

    def __init__(self):
        self.pending = ""

    def feed(self, s):
        buf = self.pending + s
        # emit the longest prefix that cannot be part of a marker
        for i in range(len(buf)):
            tail = buf[i:]
            if any(m.startswith(tail) for m in self.MARKERS):
                out = buf[:i]
                for m in self.MARKERS:
                    out = out.replace(m, "")
                self.pending = tail
                return out
        self.pending = ""
        for m in self.MARKERS:
            buf = buf.replace(m, "")
        return buf


if __name__ == "__main__":
    socket.setdefaulttimeout(30)
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
