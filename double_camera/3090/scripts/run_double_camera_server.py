from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _find_original_3090_root() -> Path:
    candidates = [
        os.environ.get("ORIGINAL_3090_ROOT", ""),
        str(Path.home() / "3090"),
        str(Path(__file__).resolve().parents[3] / "3090"),
    ]
    for item in candidates:
        if not item:
            continue
        root = Path(item).expanduser().resolve()
        if (root / "scripts" / "run_3090_server.py").exists():
            return root
    raise RuntimeError("Cannot find original 3090 folder. Set ORIGINAL_3090_ROOT.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start original 3090 server with double-camera config.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--command", default="等待双目摄像头指令")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "double_camera_3090_remote.yaml"),
    )
    args = parser.parse_args()

    original_root = _find_original_3090_root()
    os.chdir(original_root)
    if str(original_root / "src") not in sys.path:
        sys.path.insert(0, str(original_root / "src"))
    sys.argv = [
        str(original_root / "scripts" / "run_3090_server.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--command",
        args.command,
    ]
    runpy.run_path(str(original_root / "scripts" / "run_3090_server.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

