"""Back up and deploy conservative visual-landmark localization to the Pi."""

from __future__ import annotations

import os

from tools.deploy_object_search import ROOT, connect, run_checked


FILES = (
    "raspberry_pi/known_map_navigation.py",
    "raspberry_pi/visual_place_localization.py",
    "raspberry_pi/object_search_mission.py",
    "raspberry_pi/mapping/adaptive_speed.py",
    "raspberry_pi/esp32/client.py",
    "raspberry_pi/companion_service.py",
    "raspberry_pi/bluetooth_recovery.py",
    "raspberry_pi/map_camera_inspection.py",
    "raspberry_pi/scripts/robot-bluetooth-reset",
    "raspberry_pi/scripts/robot-companion-bluetooth.sudoers",
    "tests/test_visual_place_localization.py",
    "tests/test_known_map_navigation.py",
    "tests/test_companion_service.py",
    "tests/test_pi_esp32_client.py",
    "search_object.sh",
    "inspect_map.sh",
)


def main() -> None:
    client = connect("192.0.2.11", "dachuang", os.environ["PI_SSH_PASSWORD"])
    remote_root = "/home/robot/dachuang/car1.0"
    try:
        run_checked(
            client,
            "pids=$(pgrep -f '^python3 -u -m raspberry_pi.map_camera_inspection' || true); "
            "if [ -n \"$pids\" ]; then kill -INT $pids; "
            "for i in 1 2 3 4 5 6 7 8 9 10; do "
            "pgrep -f '^python3 -u -m raspberry_pi.map_camera_inspection' >/dev/null || exit 0; "
            "sleep 1; done; "
            "kill -TERM $pids; fi",
            timeout=20,
        )
        backup = run_checked(
            client,
            "cd ~/dachuang/car1.0 && stamp=$(date +%Y%m%d_%H%M%S) && "
            "backup=backups/visual_landmark_$stamp && mkdir -p $backup/raspberry_pi/mapping $backup/raspberry_pi/esp32 raspberry_pi/scripts tests && "
            "cp raspberry_pi/known_map_navigation.py raspberry_pi/visual_place_localization.py "
            "raspberry_pi/object_search_mission.py raspberry_pi/companion_service.py "
            "raspberry_pi/map_camera_inspection.py $backup/raspberry_pi/ 2>/dev/null || true; "
            "cp raspberry_pi/mapping/adaptive_speed.py $backup/raspberry_pi/mapping/ 2>/dev/null || true; "
            "cp raspberry_pi/esp32/client.py $backup/raspberry_pi/esp32/ 2>/dev/null || true; "
            "cp search_object.sh $backup/ 2>/dev/null || true; "
            "cp inspect_map.sh $backup/ 2>/dev/null || true; "
            "printf '%s' $backup",
            timeout=45,
        ).strip()
        with client.open_sftp() as sftp:
            for relative in FILES:
                sftp.put(str(ROOT / relative), f"{remote_root}/{relative}")
        output = run_checked(
            client,
            "cd ~/dachuang/car1.0 && source .venv-pi/bin/activate && "
            "bash -n inspect_map.sh && chmod +x inspect_map.sh && "
            "python3 -m compileall -q raspberry_pi && "
            "python3 -m unittest tests.test_visual_place_localization tests.test_known_map_navigation "
            "tests.test_companion_service tests.test_pi_esp32_client && "
            "sudo -S systemctl restart robot-companion.service >/dev/null 2>&1 <<< '1' && "
            "sleep 2 && systemctl is-active robot-companion.service",
            timeout=180,
        )
        print(f"PI_VISUAL_LANDMARK_DEPLOYMENT_OK backup={backup} service={output.strip()}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
