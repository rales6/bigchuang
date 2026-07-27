from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _find_windows_console_root() -> Path:
    candidates = [
        os.environ.get("WINDOWS_CAMERA_ROOT", ""),
        str(Path(__file__).resolve().parents[3] / "windows摄像头"),
    ]
    for item in candidates:
        if not item:
            continue
        root = Path(item).resolve()
        if (root / "scripts" / "run_web_console.py").exists():
            return root
    raise RuntimeError("Cannot find windows摄像头 folder. Set WINDOWS_CAMERA_ROOT.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Windows web console for double-camera mode.")
    parser.add_argument("--server", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--remote-fps", type=float, default=30.0)
    parser.add_argument("--no-auto-tunnel", action="store_true")
    args = parser.parse_args()

    root = _find_windows_console_root()
    os.chdir(root)
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    sys.argv = [
        str(root / "scripts" / "run_web_console.py"),
        "--camera-mode",
        "remote",
        "--server",
        args.server,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--remote-fps",
        str(args.remote_fps),
    ]
    if args.no_auto_tunnel:
        sys.argv.append("--no-auto-tunnel")
    runpy.run_path(str(root / "scripts" / "run_web_console.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

