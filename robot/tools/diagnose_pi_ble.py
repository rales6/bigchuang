from __future__ import annotations

import os

import paramiko


def run(client, command: str, timeout: int = 45) -> tuple[int, str]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    text = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    return code, text + error


def main() -> None:
    password = os.environ["PI_SSH_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "pi.example.invalid", username="dachuang", password=password,
        timeout=45, auth_timeout=45, banner_timeout=45,
    )
    try:
        checks = (
            "echo SHOW; bluetoothctl show; "
            "echo RFKILL; rfkill list bluetooth; "
            "echo SERVICE; systemctl is-active bluetooth; "
            f"echo LOW_LEVEL_SCAN; echo '{password}' | sudo -S timeout 12s btmgmt --index 0 find -l"
        )
        code, before = run(client, checks, timeout=35)
        print("BEFORE_RESTART")
        print(before)
        address_seen = "AA:BB:CC:DD:EE:01" in before.upper()
        if address_seen:
            print("ESP32_ADVERTISEMENT_SEEN")
            return
        restart = (
            f"echo '{password}' | sudo -S rfkill unblock bluetooth; "
            f"echo '{password}' | sudo -S systemctl restart bluetooth; sleep 3; "
            "bluetoothctl power on; "
            f"echo AFTER_SCAN; echo '{password}' | sudo -S timeout 15s btmgmt --index 0 find -l"
        )
        code, after = run(client, restart, timeout=40)
        print("AFTER_RESTART")
        print(after)
        if "AA:BB:CC:DD:EE:01" in after.upper():
            print("ESP32_ADVERTISEMENT_RECOVERED")
        else:
            print("ESP32_ADVERTISEMENT_NOT_SEEN")
    finally:
        client.close()


if __name__ == "__main__":
    main()
