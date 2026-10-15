"""离线测试脚本 — 不依赖硬件，用静态图片测试跟踪和伺服逻辑。

用法:
  python -m grab_target.test_grasp --image test_scene.jpg

如果提供了包含爪子和目标物的场景图片，程序会:
  1. 手动标记爪子 + 目标物 bbox (或通过配置文件指定)
  2. 运行双目标跟踪
  3. 输出伺服命令序列
  4. 显示叠加了 bbox 和运动方向的画面
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from grab_target.arm_calibration import ArmVisualCalibration, DEFAULT_CALIBRATION
from grab_target.dual_tracker import DualTracker
from grab_target.visual_servo import VisualServoController
from grab_target.grasp_sequence import GraspSequence, Phase


def _load_test_scene(path: str) -> np.ndarray:
    """加载测试场景图片。"""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return frame


def _simulate_claw_movement(frame: np.ndarray, bbox, left_deg, up_mm, forward_mm) -> Any:
    """模拟爪子移动后的 bbox 变化 (用于测试)。"""
    from qwen_grounded_tracker.domain import BBox

    h, w = frame.shape[:2]
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    bw = bbox.width
    bh = bbox.height

    # 模拟旋转: bbox 水平移动
    px_per_deg = 1.0 / DEFAULT_CALIBRATION.deg_per_pixel
    new_cx = cx - left_deg * px_per_deg

    # 模拟升降: bbox 垂直移动
    px_per_mm = 1.0 / DEFAULT_CALIBRATION.mm_per_pixel_vertical
    new_cy = cy - up_mm * px_per_mm

    # 模拟前伸: bbox 放大
    scale = 1.0 + forward_mm * DEFAULT_CALIBRATION.area_gain_per_forward_mm * 40
    new_bw = bw * max(0.5, min(2.0, scale))
    new_bh = bh * max(0.5, min(2.0, scale))

    return BBox(
        max(0, new_cx - new_bw / 2),
        max(0, new_cy - new_bh / 2),
        min(w, new_cx + new_bw / 2),
        min(h, new_cy + new_bh / 2),
    )


def _draw_result(frame, claw_bbox, target_bbox, servo_cmd, sequence_status) -> np.ndarray:
    """在画面上叠加跟踪结果和伺服信息。"""
    output = frame.copy()
    h, w = output.shape[:2]

    # 画爪子 bbox (蓝色)
    if claw_bbox is not None:
        cv2.rectangle(
            output,
            (int(claw_bbox.x1), int(claw_bbox.y1)),
            (int(claw_bbox.x2), int(claw_bbox.y2)),
            (255, 140, 40), 2,
        )
        cx, cy = claw_bbox.center
        cv2.circle(output, (int(cx), int(cy)), 5, (255, 80, 40), -1)
        cv2.putText(output, "CLAW", (int(claw_bbox.x1), int(claw_bbox.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 40), 2)

    # 画目标 bbox (绿色)
    if target_bbox is not None:
        cv2.rectangle(
            output,
            (int(target_bbox.x1), int(target_bbox.y1)),
            (int(target_bbox.x2), int(target_bbox.y2)),
            (40, 220, 80), 2,
        )
        cx, cy = target_bbox.center
        cv2.circle(output, (int(cx), int(cy)), 5, (40, 200, 60), -1)
        cv2.putText(output, "TARGET", (int(target_bbox.x1), int(target_bbox.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 80), 2)

    # 画连线
    if claw_bbox is not None and target_bbox is not None:
        claw_cx, claw_cy = claw_bbox.center
        target_cx, target_cy = target_bbox.center
        cv2.line(output, (int(claw_cx), int(claw_cy)),
                 (int(target_cx), int(target_cy)), (200, 200, 200), 1, cv2.LINE_AA)

    # 伺服信息
    status = sequence_status
    info_lines = [
        f"Phase: {status.phase} | Step: {status.step_label}",
        f"Action: {status.last_action} | {status.last_reason[:60]}",
    ]
    if isinstance(servo_cmd, dict):
        info_lines.append(
            f"Cmd: F={servo_cmd.get('forward_mm',0)} "
            f"U={servo_cmd.get('up_mm',0)} "
            f"L={servo_cmd.get('left_deg',0)} "
            f"C={servo_cmd.get('claw_open','-')}"
        )

    y0 = 30
    for line in info_lines:
        cv2.putText(output, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y0 += 22

    return output


def test_with_static_image(
    image_path: str,
    claw_bbox_xywh: tuple[float, float, float, float] | None = None,
    target_bbox_xywh: tuple[float, float, float, float] | None = None,
    output_dir: str = "grab_target/test_output",
) -> None:
    """用静态图片测试跟踪和伺服逻辑。"""
    from qwen_grounded_tracker.domain import BBox

    frame = _load_test_scene(image_path)
    h, w = frame.shape[:2]
    print(f"加载图片: {image_path} ({w}x{h})")

    cal = DEFAULT_CALIBRATION
    cal.camera_width = w
    cal.camera_height = h

    # 手动标记或使用默认值
    if claw_bbox_xywh is None:
        print("\n请在画面中框选 机械爪...")
        claw_roi = cv2.selectROI("Mark Claw", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Mark Claw")
        x, y, bw, bh = [int(v) for v in claw_roi]
        if bw <= 2 or bh <= 2:
            print("未标记爪子，使用默认区域")
            claw_bbox = BBox(w * 0.25, h * 0.55, w * 0.75, h * 0.90)
        else:
            claw_bbox = BBox(float(x), float(y), float(x + bw), float(y + bh))
    else:
        x, y, bw, bh = claw_bbox_xywh
        claw_bbox = BBox(x, y, x + bw, y + bh)

    if target_bbox_xywh is None:
        print("\n请在画面中框选 目标物...")
        target_roi = cv2.selectROI("Mark Target", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Mark Target")
        x, y, bw, bh = [int(v) for v in target_roi]
        if bw <= 2 or bh <= 2:
            print("未标记目标物，使用默认区域")
            target_bbox = BBox(w * 0.35, h * 0.15, w * 0.65, h * 0.40)
        else:
            target_bbox = BBox(float(x), float(y), float(x + bw), float(y + bh))
    else:
        x, y, bw, bh = target_bbox_xywh
        target_bbox = BBox(x, y, x + bw, y + bh)

    print(f"爪子: {claw_bbox.to_xywh()}")
    print(f"目标: {target_bbox.to_xywh()}")

    # 初始化
    tracker = DualTracker(cal)
    tracker.initialize_claw_only(frame, claw_bbox)
    tracker.initialize_target(frame, target_bbox)

    servo = VisualServoController(cal)
    sequence = GraspSequence("test target")
    sequence.start()
    sequence.on_qwen_done(True, True, "test target")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 模拟伺服循环
    max_steps = 15
    step = 0
    simulated_claw = claw_bbox

    print(f"\n开始模拟伺服循环 (最多 {max_steps} 步):")
    print("-" * 60)

    for step in range(max_steps):
        # 更新双目标跟踪
        result = tracker.update(frame)

        # 计算伺服命令
        servo_cmd = servo.compute(
            simulated_claw, target_bbox, frame.shape, dual_result=result,
        )

        # 打印命令
        print(f"[Step {step + 1}] {servo_cmd.state}: {servo_cmd.reason}")
        print(f"  forward={servo_cmd.forward_mm}mm, "
              f"up={servo_cmd.up_mm}mm, "
              f"left={servo_cmd.left_deg}deg, "
              f"claw={servo_cmd.claw_open}, "
              f"action={servo_cmd.action}")

        # 更新序列
        advance = sequence.update_from_servo(servo_cmd)

        # 画出当前状态
        status = sequence.status()
        viz = _draw_result(frame, simulated_claw, target_bbox, servo_cmd, status)
        cv2.imwrite(str(output_path / f"step_{step:03d}.jpg"), viz)

        servo.note_executed()

        if servo_cmd.action == "grasp":
            print(f"\n✅ 抓取!")
            break
        elif servo_cmd.action == "error":
            print(f"\n❌ 错误: {servo_cmd.reason}")
            break
        elif servo_cmd.action == "reground":
            print(f"\n🔄 需要重识别")
            break

        # 模拟爪子移动
        if servo_cmd.action == "move":
            simulated_claw = _simulate_claw_movement(
                frame, simulated_claw,
                servo_cmd.left_deg, servo_cmd.up_mm, servo_cmd.forward_mm,
            )
            # 更新 tracker 中的爪子 bbox
            tracker.claw.bbox = simulated_claw
            tracker.claw.tracker = tracker._create_csrt()
            tracker.claw.tracker.init(frame, simulated_claw.to_xywh())

        if servo.attempts_exceeded:
            print(f"\n⚠ 超出最大尝试次数")
            break

    print("-" * 60)
    print(f"完成 {step + 1} 步，输出图片保存在 {output_path}/")
    print(f"最终状态: {sequence.status().step_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Visual Servo Grasp")
    parser.add_argument("--image", required=True, help="场景图片路径")
    parser.add_argument("--claw", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        help="爪子 bbox (xywh)")
    parser.add_argument("--target", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        help="目标物 bbox (xywh)")
    parser.add_argument("--output", default="grab_target/test_output")
    args = parser.parse_args()

    claw_bbox = tuple(args.claw) if args.claw else None
    target_bbox = tuple(args.target) if args.target else None

    try:
        test_with_static_image(
            args.image,
            claw_bbox_xywh=claw_bbox,
            target_bbox_xywh=target_bbox,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("\n测试取消")


if __name__ == "__main__":
    main()
