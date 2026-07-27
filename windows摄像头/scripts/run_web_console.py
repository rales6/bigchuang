from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from qwen_grounded_tracker.web_client_app_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
