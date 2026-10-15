#!/usr/bin/env python3
"""Phone bridge that launches the independent relationship-map real-car path."""

from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
import time

CAR_ROOT = Path(
    os.environ.get("CAR_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(CAR_ROOT) not in sys.path:
    sys.path.insert(0, str(CAR_ROOT))

from raspberry_pi import companion_service as protected


RELEASE_DIR = Path(__file__).resolve().parent
REAL_SEARCH = RELEASE_DIR / "natural_search_relationship.sh"


class RelationshipCompanionService(protected.CompanionService):
    def start_search(self, instruction: str) -> dict[str, object]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("任务指令不能为空")
        if not REAL_SEARCH.is_file():
            raise RuntimeError(f"新版实车入口不存在：{REAL_SEARCH}")
        if self.runner.snapshot()["running"]:
            self.runner.stop()
        self.emergency_stop = False
        protected.DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        planning_decision = {
            "instruction": instruction,
            "status": "planning",
            "target": {},
            "anchor": {},
            "steps": [
                {
                    "id": f"step_{index:02d}",
                    "action": action,
                    "success": protected.ACTION_LABELS[action],
                }
                for index, action in enumerate(protected.ACTION_LABELS, start=1)
            ],
        }
        protected.DECISION_PATH.write_text(
            json.dumps(planning_decision, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = self.start_managed_service(
            "object_search", [str(REAL_SEARCH), instruction]
        )
        self.reply = "已接收指令，正在使用新版语义区域规划实车路线"
        self.reply_id = f"task-{int(time.time() * 1000)}"
        return result


def run(host: str, port: int, config_path: Path) -> None:
    service = RelationshipCompanionService(config_path)
    server = ThreadingHTTPServer((host, int(port)), protected._handler(service))
    print(
        f"Robot Companion relationship API: http://{host}:{port}; "
        f"real_search={REAL_SEARCH}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        service.runner.stop()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RobotCompanionAndroid relationship-map real-car bridge"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--config", type=Path, default=protected.CONFIG_PATH)
    args = parser.parse_args()
    run(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
