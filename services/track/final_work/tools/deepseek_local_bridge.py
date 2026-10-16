#!/usr/bin/env python3
"""Loopback-only DeepSeek bridge for browsers that block cross-origin calls.

The bridge keeps no configuration and writes no request bodies or API keys to
disk.  It accepts calls only from the robot UI origins listed with --origin.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib import error as urlerror
from urllib import request as urlrequest


MAX_BODY = 128 * 1024
UPSTREAM = "https://api.deepseek.com/chat/completions"


class Handler(BaseHTTPRequestHandler):
    allowed_origins: set[str] = set()

    def log_message(self, format: str, *args: object) -> None:
        # Avoid access logs entirely so authorization metadata can never leak.
        return

    def _origin(self) -> str:
        return self.headers.get("Origin", "")

    def _cors(self) -> None:
        origin = self._origin()
        if origin in self.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if self._origin() not in self.allowed_origins:
            self._json(403, {"error": "origin not allowed"})
            return
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        if self._origin() not in self.allowed_origins:
            self._json(403, {"error": "origin not allowed"})
            return
        authorization = self.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or len(authorization) > 600:
            self._json(401, {"error": "missing API key"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": "invalid request size"})
            return
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid JSON"})
            return
        upstream = urlrequest.Request(
            UPSTREAM,
            data=body,
            method="POST",
            headers={"Authorization": authorization, "Content-Type": "application/json"},
        )
        try:
            with urlrequest.urlopen(upstream, timeout=95) as response:
                value = response.read(MAX_BODY)
                status = response.status
        except urlerror.HTTPError as exc:
            value = exc.read(MAX_BODY)
            status = exc.code
        except Exception as exc:
            self._json(502, {"error": f"DeepSeek connection failed: {type(exc).__name__}"})
            return
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek user-device loopback bridge")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--origin",
        action="append",
        default=["http://192.0.2.20:8030"],
        help="Allowed robot UI origin; repeat for more origins",
    )
    args = parser.parse_args()
    Handler.allowed_origins = {item.rstrip("/") for item in args.origin}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"DeepSeek 本机桥已启动：http://127.0.0.1:{args.port}")
    print("密钥只从浏览器转发到 DeepSeek，不会保存或打印。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
