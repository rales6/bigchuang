"""统计 web_app 日志中 BLE 指令从入队到写入的延迟。"""
import glob
import json

events = []
for path in glob.glob("/home/robot/grab_visual/logs/web_app.jsonl"):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
events.sort(key=lambda e: e.get("time_unix", 0))

ble = [
    e for e in events
    if e.get("event") in ("ble_queued", "ble_written",
                          "ble_write_error", "ble_write_timeout",
                          "ble_rejected", "ble_connected", "ble_disconnected")
][-120:]

queued = {}
for e in ble:
    ev = e["event"]
    if ev == "ble_queued":
        queued[e.get("command", "")] = e
    elif ev == "ble_written" and e.get("command") in queued:
        q = queued.pop(e["command"])
        print("{} | {} -> {} = {}ms".format(
            e["ts"][11:19], q["ts"][11:19], e["ts"][11:19],
            round((e["time_unix"] - q["time_unix"]) * 1000),
        ))
    elif ev in ("ble_write_error", "ble_write_timeout", "ble_rejected"):
        print("{} | {}: {}".format(e["ts"][11:19], ev, str(e.get("command"))[:40]))
    elif ev in ("ble_connected", "ble_disconnected"):
        print("{} | {}".format(e["ts"][11:19], ev))
