"""物品夹取仿真样例。

网页保持打开时运行本模块。脚本会选择夹取任务、导航到第一个物品前方，
读取前置摄像头检测结果，再按照项目真实的六关节脉宽协议完成下降和夹取。
"""

from __future__ import annotations

import argparse
import math
import time

from car_sim.virtual_hardware import SimulatedCameraClient, SimulatedEsp32Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="网页仿真物品夹取样例")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=25.0)
    return parser


def wait_for_detection(
    camera: SimulatedCameraClient,
    timeout_s: float,
) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        detections = camera.detections()
        if detections:
            return min(detections, key=lambda item: item["distance_m"])
        time.sleep(0.15)
    raise RuntimeError("摄像头在限定时间内没有检测到物品")


def main() -> None:
    args = build_parser().parse_args()
    car = SimulatedEsp32Client(base_url=args.base_url)
    camera = SimulatedCameraClient(base_url=args.base_url)
    car.heartbeat()
    car.start()
    car.select_task("pickup")

    scene_response = car.http.request("/api/scene")
    scene = scene_response.get("scene", scene_response)
    items = [item for item in scene.get("items", []) if not item.get("held")]
    if not items:
        raise RuntimeError("场景中没有物品；请在“场景布置 → 布置 → 放置物品”中添加")

    status = car.query_status()
    pose = status["ground_truth_pose"]
    item = items[0]
    dx = float(item["x"]) - float(pose["x_m"])
    dy = float(item["y"]) - float(pose["y_m"])
    distance = max(0.001, math.hypot(dx, dy))
    approach_distance = 0.9
    approach_x = float(item["x"]) - dx / distance * approach_distance
    approach_y = float(item["y"]) - dy / distance * approach_distance

    print(
        "夹取任务开始：物品 #{}，导航到 ({:.2f}, {:.2f})".format(
            item["id"], approach_x, approach_y
        )
    )
    car.goto(approach_x, approach_y)

    deadline = time.monotonic() + args.timeout
    detection = None
    while time.monotonic() < deadline:
        detections = camera.detections()
        if detections:
            candidate = min(detections, key=lambda item: item["distance_m"])
            if candidate["distance_m"] <= 1.2:
                detection = candidate
                break
        time.sleep(0.15)
    if detection is None:
        raise RuntimeError("小车未能在限定时间内到达可夹取位置")

    bearing_deg = math.degrees(float(detection["bearing_rad"]))
    base_pulse = max(500, min(2500, round(1500 + bearing_deg * 10)))
    print(
        "摄像头锁定：bbox={}，距离={:.2f}m，偏角={:.1f}°".format(
            detection["bbox"],
            detection["distance_m"],
            bearing_deg,
        )
    )

    try:
        car.set_arm_joints(
            [(0, 1500), (1, 1700), (2, 2000), (3, 1100), (4, 1500), (5, 1200)],
            duration_ms=900,
        )
        time.sleep(1.0)
        car.set_arm_joints([(0, base_pulse)], duration_ms=500)
        time.sleep(0.6)
        car.set_arm_joints(
            [(1, 1200), (2, 2100), (3, 1000), (4, 1500)],
            duration_ms=900,
        )
        time.sleep(1.0)
        car.set_arm_joints([(5, 1500)], duration_ms=500)
        time.sleep(0.7)
        car.set_arm_joints(
            [(1, 1600), (2, 1700), (3, 1350)],
            duration_ms=900,
        )
        time.sleep(1.0)
        result = car.query_status()
        if result.get("grasped_item_id") is None:
            raise RuntimeError("机械臂动作已经执行，但物品未被夹住")
        print("夹取序列完成，关节位置：", result["joint_positions"])
        print("已夹取物品：#{}".format(result["grasped_item_id"]))
    finally:
        car.arm_stop()
        car.stop()
        car.close(send_cancel=False)


if __name__ == "__main__":
    main()
