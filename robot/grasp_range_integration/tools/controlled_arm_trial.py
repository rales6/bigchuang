"""Explicitly authorized arm-only trial through the existing Pi web BLE owner."""
from __future__ import annotations

import argparse
import json
import time
from urllib import request as urllib_request


def request_json(base: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib_request.urlopen(req, timeout=8) as response:
        value = json.loads(response.read().decode("utf-8", "replace"))
    if not value.get("ok", True):
        raise RuntimeError(str(value.get("error") or value))
    return value


def raw(base: str, command: str) -> None:
    result = request_json(base, "/api/arm/raw", {"ascii": command})
    if result.get("result") != "OK":
        raise RuntimeError(f"BLE did not accept {command}: {result}")
    print(json.dumps({"sent": command, "result": "OK", "time": time.time()}))


def heartbeat(base: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(0.35, max(0.0, deadline - time.monotonic())))
        raw(base, "#PING!")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--s0", type=int, required=True)
    parser.add_argument("--s1", type=int, required=True)
    parser.add_argument("--s2", type=int, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "ARM_ONLY_OUTSIDE_HULL_AUTHORIZED":
        raise SystemExit("confirmation token rejected")
    limits = {"s0": (500, 2500), "s1": (800, 1700), "s2": (1500, 2200)}
    for name in limits:
        value = int(getattr(args, name))
        low, high = limits[name]
        if not low <= value <= high:
            raise SystemExit(f"{name}={value} outside [{low}, {high}]")
    status = request_json(args.base_url, "/api/status")
    if not status.get("ble"):
        raise SystemExit("BLE is not connected")
    if (status.get("grasp") or {}).get("active"):
        raise SystemExit("another grasp task is active")
    target = f"#ARMG{args.s0:04d}{args.s1:04d}{args.s2:04d}!"
    try:
        raw(args.base_url, "#HOME!")
        heartbeat(args.base_url, 4.6)
        raw(args.base_url, target)
        heartbeat(args.base_url, 4.7)
        raw(args.base_url, "#ARMC0!")
        heartbeat(args.base_url, 1.4)
        raw(args.base_url, "#ARMG150017002000!")
        heartbeat(args.base_url, 4.7)
        print(json.dumps({
            "ok": True,
            "arm_only": True,
            "chassis_commands_sent": False,
            "target": target,
            "claw": "closed",
            "final_pose": "joint_home_with_claw_held",
        }))
        return 0
    except Exception:
        try:
            request_json(args.base_url, "/api/arm/stop", {})
        finally:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
