from __future__ import annotations

import json
import os
import shlex
import paramiko


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("198.51.100.20", port=2020, username="sjtu", password=os.environ["RTX_SSH_PASSWORD"], timeout=45, auth_timeout=45, banner_timeout=45)
    commands = [
        "去柜子边上找一个红色瓶子",
        "去马路上找一个红色物体",
        "去簸箕旁找一个红色物品",
        "簸箕在地图的右下角",
        "\u53bb\u67dc\u5b50\u8fb9\u627e\u4e00\u4e2a\u7ea2\u8272\u5c0f\u7269\u5757",
    ]
    expected = ["executable", "rejected", "needs_clarification", "executable", "executable"]
    actual = []
    for instruction in commands:
        payload = json.dumps({"instruction": instruction}, ensure_ascii=False)
        code_text = (
            "import urllib.request; "
            f"b={payload!r}.encode('utf-8'); "
            "r=urllib.request.Request('http://127.0.0.1:8011/mission/command',data=b,headers={'Content-Type':'application/json'},method='POST'); "
            "print(urllib.request.urlopen(r,timeout=150).read().decode('utf-8'))"
        )
        command = "/usr/bin/python3 -c " + shlex.quote(code_text)
        _, out, err = c.exec_command(command, timeout=180)
        code = out.channel.recv_exit_status()
        text = out.read().decode("utf-8", errors="replace").strip()
        error = err.read().decode("utf-8", errors="replace").strip()
        if code:
            c.close()
            raise RuntimeError(error or text)
        value = json.loads(text)
        decision = value["decision"]
        actual.append(decision["status"])
        if instruction.endswith("\u7ea2\u8272\u5c0f\u7269\u5757"):
            mode = (decision.get("target") or {}).get("detection_mode")
            if mode != "qwen_fallback":
                c.close()
                raise RuntimeError(f"expected qwen_fallback, got {mode!r}")
        print(json.dumps({
            "instruction": instruction,
            "status": decision["status"],
            "reason": decision.get("reason"),
            "question": decision.get("clarification_question"),
            "source": decision.get("source"),
        }, ensure_ascii=False))
    state_payload = json.dumps(
        {
            "instruction": "去柜子边上找一个红色瓶子",
            "target": {"class_name": "bottle", "color": "red"},
            "anchor": {"landmark_id": "object_10_1"},
            "current_view_index": 0,
            "total_views": 6,
            "remaining_view_indices": [1, 2, 3, 4, 5],
            "latest_event": {"type": "observation_complete"},
            "detections": [],
            "attempts": 1,
        },
        ensure_ascii=False,
    )
    code_text = (
        "import urllib.request; "
        f"b={state_payload!r}.encode('utf-8'); "
        "r=urllib.request.Request('http://127.0.0.1:8011/mission/next',data=b,headers={'Content-Type':'application/json'},method='POST'); "
        "print(urllib.request.urlopen(r,timeout=150).read().decode('utf-8'))"
    )
    _, out, err = c.exec_command("/usr/bin/python3 -c " + shlex.quote(code_text), timeout=180)
    code = out.channel.recv_exit_status()
    text = out.read().decode("utf-8", errors="replace").strip()
    error = err.read().decode("utf-8", errors="replace").strip()
    if code:
        c.close()
        raise RuntimeError(error or text)
    directive = json.loads(text)["directive"]
    print(json.dumps({"mission_next": directive}, ensure_ascii=False))
    if directive["action"] != "navigate_next_view" or directive["terminal"]:
        c.close()
        raise RuntimeError(f"unexpected mission directive: {directive}")
    c.close()
    if actual != expected:
        raise RuntimeError(f"live decisions mismatch: {actual} != {expected}")
    print("LIVE_MISSION_API_VALIDATION_OK")


if __name__ == "__main__":
    main()
