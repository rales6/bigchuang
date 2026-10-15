"""Small rotating JSONL event logger shared by the Pi and 3090 processes."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"encoding": "hex", "data": bytes(value).hex()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


class EventLogger:
    """Write UTF-8 JSON lines, rotating at 10 MiB with five backups."""

    def __init__(self, component: str, filename: str) -> None:
        log_dir = Path(os.environ.get("GRAB_LOG_DIR", "logs")).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / filename
        self.component = component
        self._throttle: dict[str, float] = {}
        self._lock = threading.Lock()
        name = f"grab.{component}.{self.path.resolve()}"
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = RotatingFileHandler(
                self.path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "time_unix": round(time.time(), 3),
            "component": self.component,
            "event": event,
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        self._logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))

    def throttled(self, key: str, every_s: float, event: str, **fields: Any) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._throttle.get(key, 0.0) < every_s:
                return
            self._throttle[key] = now
        self.event(event, **fields)
