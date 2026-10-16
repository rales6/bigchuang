from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit natural-language object search to guarded 3090 planner")
    parser.add_argument("instruction")
    parser.add_argument("--instruction-file", default="", help="UTF-8 text file")
    parser.add_argument("--3090-url", dest="gpu_url", default="http://192.0.2.20:8011")
    parser.add_argument("--output", default="search_runs/latest_decision.json")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    instruction = (
        Path(args.instruction_file).read_text(encoding="utf-8").strip()
        if args.instruction_file else args.instruction
    )
    if not instruction:
        raise RuntimeError("instruction must not be empty")
    body = json.dumps({"instruction": instruction}, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        args.gpu_url.rstrip("/") + "/mission/command",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"3090自然语言规划请求失败：{exc}") from exc
    if not isinstance(payload, dict) or not payload.get("ok", False):
        raise RuntimeError(str(payload.get("error") if isinstance(payload, dict) else payload))
    decision = payload["decision"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    status = decision.get("status")
    if status == "needs_clarification":
        print("需要补充信息：" + str(decision.get("clarification_question") or ""), flush=True)
        raise SystemExit(3)
    if status != "executable":
        print("指令不能执行：" + str(decision.get("reason") or ""), flush=True)
        raise SystemExit(4)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
