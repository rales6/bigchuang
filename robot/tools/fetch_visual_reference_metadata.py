"""Download read-only inspection/semantic metadata for visual calibration."""

from __future__ import annotations

import os
from pathlib import Path

from tools.deploy_object_search import connect


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "search_runs" / "visual_reference_metadata"


def main() -> None:
    client = connect("192.0.2.11", "dachuang", os.environ["PI_SSH_PASSWORD"])
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        remote_root = "/home/robot/dachuang/car1.0"
        with client.open_sftp() as sftp:
            for remote, local in (
                (f"{remote_root}/maps/ble_scene_17_semantics.json", OUTPUT / "semantics.json"),
                (f"{remote_root}/maps/ble_scene_17_semantics_qwen.json", OUTPUT / "semantics_qwen.json"),
                (f"{remote_root}/inspections/ble_scene_17/20260813_153031/inspection_manifest.json", OUTPUT / "manifest_previous.json"),
                (f"{remote_root}/inspections/ble_scene_17/20260813_163201/inspection_manifest.json", OUTPUT / "manifest_latest.json"),
                (f"{remote_root}/search_runs/visual_tokens_qwen_calibration.json", OUTPUT / "visual_tokens_qwen_calibration.json"),
            ):
                try:
                    sftp.get(remote, str(local))
                except FileNotFoundError:
                    pass
        print(OUTPUT)
    finally:
        client.close()


if __name__ == "__main__":
    main()
