"""Apply the camera ownership environment to every grab web restart."""
import os
from pathlib import Path
import sys

from .camera_proxy import camera_environment


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Missing grab web entry point')
    script = Path(sys.argv[1])
    if not script.is_absolute() or not script.is_file():
        raise SystemExit('Invalid grab web entry point')
    # Re-exec so Python loads camera_bootstrap/sitecustomize before web_app
    # imports subprocess/OpenCV. Manual/watchdog starts get the same adapter
    # as pursuit recovery, hull alignment and placement preview restores.
    os.execve(sys.executable, [sys.executable, str(script), *sys.argv[2:]],
              camera_environment())


if __name__ == '__main__':
    main()
