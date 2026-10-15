"""Read-only dual-camera/DeepSeek check of current lidar pose candidates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from raspberry_pi.config import LidarConfig
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.map_camera_inspection import CameraCapture

from deepseek_visual_localization import (
    DeepSeekVisionClient, candidate_scene_descriptions, load_config,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    car_root = root.parent
    output = car_root / "search_runs" / "phase1" / ("pose_probe_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(
        os.environ.get("DEEPSEEK_VISION_CONFIG", car_root / "secrets" / "deepseek_vision.json"),
        module_directory=root,
    )
    if not config.configured:
        raise RuntimeError("DeepSeek Vision 配置不可用；未拍照或发送任何图片")
    CameraCapture._stable_usb_capture_nodes = staticmethod(
        lambda: ["/dev/video2", "/dev/video0"]
    )
    camera = CameraCapture("opencv", 2, 1280, 720, 15.0, dual_usb_cameras=True)
    lidar = N10LidarDriver(LidarConfig(port="/dev/ttyACM0", baudrate=230400))
    try:
        left = output / "current_left.jpg"
        capture = camera.capture(left)
        second_name = capture.get("camera_2_image")
        if not second_name:
            raise RuntimeError("双摄没有同时生成左右图像；不进行 DeepSeek 消歧")
        right = left.with_name(str(second_name))
        if not right.is_file() or right == left:
            raise RuntimeError("双摄没有同时生成左右图像；不进行 DeepSeek 消歧")
        known_map = KnownOccupancyMap.load(root / "ble_scene_17.pgm", root / "ble_scene_17.yaml", clearance_m=0.30)
        localizer = ScanMapLocalizer(known_map, lidar_offset_x_m=0.14, lidar_offset_y_m=0.0)
        scan = next(iter(lidar.scans()))
        candidates = localizer.global_localization_candidates(
            scan, maximum=config.maximum_candidates
        )
        if not candidates:
            raise RuntimeError("当前雷达没有给出可用候选；未调用 DeepSeek")
        semantic = json.loads(config.semantic_map.read_text(encoding="utf-8"))
        descriptions = candidate_scene_descriptions(candidates, semantic, config)
        ranked = DeepSeekVisionClient(config).rank(left, right, descriptions)
        reliable = bool(
            ranked.selected_candidate_id != "uncertain"
            and ranked.confidence >= config.minimum_confidence
            and ranked.margin >= config.minimum_margin
            and ranked.consistent_repeats == config.repeat_count
        )
        report = {
            "motor_link_created": False,
            "left_image": str(left),
            "right_image": str(right),
            "camera_count": capture.get("camera_count"),
            "candidates": [
                {
                    "id": f"C{i:02d}",
                    "x_m": round(item.pose.x_m, 3),
                    "y_m": round(item.pose.y_m, 3),
                    "yaw_deg": round(math.degrees(item.pose.yaw_rad), 1),
                    "rmse_m": round(item.rmse_m, 3),
                    "inlier_ratio": round(item.inlier_ratio, 3),
                    "reliable_lidar": bool(item.reliable),
                }
                for i, item in enumerate(candidates, 1)
            ],
            "deepseek": ranked.to_dict(),
            "accepted": reliable,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        lidar.close()
        close_camera = getattr(camera, "close", None)
        if close_camera is not None:
            close_camera()


if __name__ == "__main__":
    main()
