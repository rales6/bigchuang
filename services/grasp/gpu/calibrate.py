"""机械臂-摄像头视觉标定程序。

在抓取前需运行此程序，标定像素→机械臂运动的映射关系。

标定流程:
  1. 机械臂归位
  2. 拍照 → 记录爪子 bbox
  3. 旋转 10° → 拍照 → 计算 deg_per_pixel
  4. 升降 15mm → 拍照 → 计算 mm_per_pixel_vertical
  5. 前伸 20mm → 拍照 → 计算 area_gain_per_forward_mm
  6. 保存 calibration JSON

用法 (在 3090 上):
  python -m grab_target.calibrate \
    --output grab_target/arm_visual_calibration.json \
    --camera-id 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _get_claw_bbox_manual(frame: np.ndarray, label: str = "claw") -> Any | None:
    """手动标记爪子 bbox (用 OpenCV selectROI)。"""
    from qwen_grounded_tracker.domain import BBox

    cv2.namedWindow(f"Select {label}", cv2.WINDOW_NORMAL)
    print(f"\n请在画面中框选 {label}，按 Enter 确认，按 C 跳过")

    roi = cv2.selectROI(f"Select {label}", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select {label}")

    x, y, w, h = [int(v) for v in roi]
    if w <= 2 or h <= 2:
        return None
    return BBox(float(x), float(y), float(x + w), float(y + h))


def calibrate(
    camera_id: int = 0,
    output_path: str = "grab_target/arm_visual_calibration.json",
) -> dict[str, Any]:
    """运行标定流程。

    重要: 此程序假设你可以在树莓派上手动发送 ASCII 命令控制机械臂。
    或者通过 3090 的 WebSocket 发送命令。
    """
    from grab_target.arm_calibration import ArmVisualCalibration

    cal = ArmVisualCalibration()

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")

    print("=" * 60)
    print("机械臂视觉标定程序")
    print("=" * 60)
    print()
    print("标定前请确保:")
    print("  1. 机械臂已归位 (home position)")
    print("  2. 摄像头能看到机械爪")
    print("  3. 可以通过 BLE 发送 ASCII 命令控制机械臂")
    print()
    print("机械臂归位参数:")
    print(f"  forward={cal.arm_home_forward_mm}mm")
    print(f"  up={cal.arm_home_up_mm}mm")
    print(f"  yaw={cal.arm_home_yaw_deg}°")
    print()

    results: dict[str, Any] = {}

    # ── Step 1: 记录归位位置 ───────────────────────────────
    input("1/4: 机械臂归位后按 Enter 拍照...")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("拍照失败")

    claw_home = _get_claw_bbox_manual(frame, "爪子 in home position")
    if claw_home is None:
        print("警告: 未标记爪子，使用画面中心下方默认区域")
        h, w = frame.shape[:2]
        from qwen_grounded_tracker.domain import BBox
        claw_home = BBox(w * 0.3, h * 0.5, w * 0.7, h * 0.85)
    results["claw_home_center"] = claw_home.center

    # ── Step 2: 旋转标定 ──────────────────────────────────
    print(f"\n2/4: 旋转标定")
    print(f"请发送命令: #ARML10! (左转 10°)")
    input("发送后等待机械臂就位，按 Enter 拍照...")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("拍照失败")

    claw_left = _get_claw_bbox_manual(frame, "爪子 after left 10deg")
    if claw_left is not None:
        dx = claw_home.center[0] - claw_left.center[0]  # 正=左移
        deg_per_pixel = 10.0 / max(abs(dx), 1.0)
        cal.deg_per_pixel = round(deg_per_pixel, 4)
        print(f"  像素位移: {dx:.1f}px → deg_per_pixel = {cal.deg_per_pixel} deg/px")
        results["deg_per_pixel_raw"] = cal.deg_per_pixel

    # 归位
    print(f"请发送命令: #ARMR10! (右转 10° 归位)")
    input("发送后等待就位，按 Enter 继续...")

    # ── Step 3: 升降标定 ──────────────────────────────────
    print(f"\n3/4: 升降标定")
    print(f"请发送命令: #ARMU15! (上升 15mm)")
    input("发送后等待机械臂就位，按 Enter 拍照...")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("拍照失败")

    claw_up = _get_claw_bbox_manual(frame, "爪子 after up 15mm")
    if claw_up is not None:
        dy = claw_home.center[1] - claw_up.center[1]  # 上升=画面中y减小
        mm_per_pixel = 15.0 / max(abs(dy), 1.0)
        cal.mm_per_pixel_vertical = round(mm_per_pixel, 4)
        print(f"  像素位移: {dy:.1f}px → mm_per_pixel = {cal.mm_per_pixel_vertical} mm/px")
        results["mm_per_pixel_raw"] = cal.mm_per_pixel_vertical

    # 归位
    print(f"请发送命令: #ARMD15! (下降 15mm 归位)")
    input("发送后等待就位，按 Enter 继续...")

    # ── Step 4: 前伸标定 ──────────────────────────────────
    print(f"\n4/4: 前伸标定")
    print(f"请发送命令: #ARMF20! (前伸 20mm)")
    input("发送后等待机械臂就位，按 Enter 拍照...")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("拍照失败")

    claw_fwd = _get_claw_bbox_manual(frame, "爪子 after forward 20mm")
    if claw_fwd is not None and claw_home is not None:
        home_area = claw_home.width * claw_home.height
        fwd_area = claw_fwd.width * claw_fwd.height
        if home_area > 0:
            area_increase = (fwd_area - home_area) / home_area
            area_gain_per_mm = area_increase / 20.0
            cal.area_gain_per_forward_mm = round(area_gain_per_mm, 6)
            print(f"  面积变化: {home_area:.0f} → {fwd_area:.0f} (+{area_increase:.1%})")
            print(f"  area_gain_per_forward_mm = {cal.area_gain_per_forward_mm}")
            results["area_gain_per_mm_raw"] = cal.area_gain_per_forward_mm

    # 归位
    print(f"请发送命令: #ARMB20! (后退 20mm 归位)")
    input("发送后等待就位，按 Enter 继续...")

    # ── 保存 ──────────────────────────────────────────────
    cap.release()

    print("\n" + "=" * 60)
    print("标定结果")
    print("=" * 60)
    print(f"  deg_per_pixel:          {cal.deg_per_pixel}")
    print(f"  mm_per_pixel_vertical:  {cal.mm_per_pixel_vertical}")
    print(f"  area_gain_per_forward_mm: {cal.area_gain_per_forward_mm}")
    print()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cal.save(output)
    print(f"标定参数已保存到: {output}")
    print()

    # 打印可导入到 config 的 JSON 片段
    print("可在启动脚本中使用的路径:")
    print(f"  --calibration {output.resolve()}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm Visual Calibration")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--output", default="grab_target/arm_visual_calibration.json")
    args = parser.parse_args()

    try:
        calibrate(camera_id=args.camera_id, output_path=args.output)
    except KeyboardInterrupt:
        print("\n标定已取消")
    except Exception as exc:
        print(f"标定失败: {exc}")
        raise


if __name__ == "__main__":
    main()
