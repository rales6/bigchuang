# -*- coding: utf-8 -*-
"""Per-run logging helpers for the Raspberry Pi pursuit process.

Each invocation gets an immutable directory instead of overwriting the previous
CSV.  stdout/stderr are tee'd to ``console.log`` so camera, Qwen, network and
controller messages stay together with the telemetry.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def _instruction_slug(text: str, limit: int = 28) -> str:
    chars: list[str] = []
    previous_separator = False
    for char in text.strip():
        if char.isalnum():
            chars.append(char)
            previous_separator = False
        elif not previous_separator:
            chars.append("_")
            previous_separator = True
        if len(chars) >= limit:
            break
    slug = "".join(chars).strip("_")
    return slug or "target"


class _TeeTextIO:
    """Small line-buffered stdout/stderr tee used by long-running field runs."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            terminal_result = self._terminal.write(text)
            self._log_file.write(text)
            self._log_file.flush()
        return terminal_result if isinstance(terminal_result, int) else len(text)

    def flush(self) -> None:
        with self._lock:
            self._terminal.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._terminal.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._terminal, "encoding", None)


class RunLogBundle:
    """Owns one run directory, metadata, console tee and telemetry path."""

    def __init__(self, run_dir: Path, metadata: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.console_path = run_dir / "console.log"
        self.telemetry_path = run_dir / "telemetry.csv"
        self.metadata_path = run_dir / "run.json"
        self.events_path = run_dir / "events.jsonl"
        self._metadata = metadata
        self._console_file: TextIO | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._closed = False
        self._write_metadata()
        atexit.register(self.close)

    @classmethod
    def create(
        cls,
        log_root: str | Path,
        instruction: str,
        mode: str,
        argv: list[str],
        arguments: dict[str, Any],
    ) -> "RunLogBundle":
        root = Path(log_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()
        stem = f"{started:%Y%m%d_%H%M%S}_{mode}_{_instruction_slug(instruction)}"
        run_dir = root / stem
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = root / f"{stem}_{suffix}"
        run_dir.mkdir(parents=True)

        resolved = run_dir.resolve()
        (root / "LATEST.txt").write_text(str(resolved) + "\n", encoding="utf-8")
        # Linux operator convenience. LATEST.txt remains the portable fallback.
        latest = root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name, target_is_directory=True)
        except OSError:
            pass

        metadata: dict[str, Any] = {
            "schema_version": 1,
            "status": "starting",
            "started_at": started.isoformat(timespec="milliseconds"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "mode": mode,
            "instruction": instruction,
            "argv": argv,
            "arguments": arguments,
            "run_dir": str(resolved),
        }
        return cls(run_dir, metadata)

    def install_console_tee(self) -> None:
        if self._console_file is not None:
            return
        self._console_file = self.console_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeTextIO(sys.stdout, self._console_file)
        sys.stderr = _TeeTextIO(sys.stderr, self._console_file)

    def choose_telemetry_path(self, override: str | None) -> Path:
        if not override:
            return self.telemetry_path
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["telemetry_override"] = str(path.resolve())
        self._write_metadata()
        return path

    def update(self, **fields: Any) -> None:
        self._metadata.update(fields)
        self._write_metadata()

    def event(self, event: str, **fields: Any) -> None:
        value = {"at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                 "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def finalize(self, status: str, **fields: Any) -> None:
        self._metadata.update(fields)
        self._metadata["status"] = status
        self._metadata["ended_at"] = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 进程若在摄像头/雷达初始化阶段异常退出，main 的 finally 尚未建立；
        # atexit 仍会把 run.json 从 starting/running 改成明确的异常结束状态。
        if "ended_at" not in self._metadata:
            self._metadata["status"] = "process_exit"
            self._metadata["ended_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            self._write_metadata()
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        if self._console_file is not None:
            self._console_file.flush()
            self._console_file.close()
