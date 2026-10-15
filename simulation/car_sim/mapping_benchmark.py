"""使用网页雷达数据对项目现有 LidarSlam 做一轮可重复的建图联调。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from raspberry_pi.config import LidarConfig, MappingConfig
from raspberry_pi.mapping import LidarSlam

from .mapping_report import generate_mapping_report
from .virtual_hardware import (
    SimulatedEsp32Client,
    SimulatedLidarDriver,
    SimulatorUnavailable,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="网页仿真建图基准测试")
    parser.add_argument("--scans", type=int, default=160, help="采集扫描帧数")
    parser.add_argument(
        "--speed",
        type=int,
        default=350,
        help="直行速度 mm/s，范围 50..550",
    )
    parser.add_argument(
        "--turn-speed",
        type=int,
        default=2500,
        help="原地转向速度 mrad/s，范围 300..3500",
    )
    parser.add_argument(
        "--output",
        default="car_sim/output/benchmark_map",
        help="地图输出前缀，必须位于 car_sim 内",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成 HTML、JSON 和逐帧 CSV 分析报告",
    )
    return parser


def drive_for_scan(
    client: SimulatedEsp32Client,
    index: int,
    speed_mm_s: int = 350,
    turn_speed_mrad_s: int = 2500,
) -> None:
    """矩形路线；速度与项目自主建图的正常巡航/转向范围一致。"""
    phase = index % 94
    if phase < 40:
        client.set_twist(speed_mm_s, 0, 450)
    elif phase < 47:
        client.set_twist(0, turn_speed_mrad_s, 450)
    elif phase < 87:
        client.set_twist(speed_mm_s, 0, 450)
    else:
        client.set_twist(0, turn_speed_mrad_s, 450)


def main() -> None:
    args = build_parser().parse_args()
    if not 50 <= args.speed <= 550:
        raise SystemExit("--speed 必须在 50..550 mm/s 之间")
    if not 300 <= args.turn_speed <= 3500:
        raise SystemExit("--turn-speed 必须在 300..3500 mrad/s 之间")
    output = Path(args.output).resolve()
    car_sim_root = Path(__file__).resolve().parent
    try:
        output.relative_to(car_sim_root)
    except ValueError as exc:
        raise SystemExit("为了隔离测试数据，--output 必须位于 car_sim 目录内") from exc
    output.parent.mkdir(parents=True, exist_ok=True)

    client = SimulatedEsp32Client(base_url=args.base_url)
    lidar = SimulatedLidarDriver(base_url=args.base_url)
    slam = LidarSlam(
        MappingConfig(
            resolution_m=0.05,
            width_cells=240,
            height_cells=240,
            min_match_points=25,
            map_max_linear_speed_m_s=0.50,
        ),
        LidarConfig(
            min_distance_m=0.12,
            max_distance_m=8.0,
            clockwise=False,
            motor_control=False,
            min_scan_points=25,
        ),
    )
    accepted = 0
    integrated = 0
    errors = []
    scan_samples = []
    started_at = time.monotonic()
    client.heartbeat()
    client.select_task("mapping")
    client.set_pose(1.0, 1.0, 0.0)
    client.reset_map()
    client.start()
    client.set_twist(0, 0, 450)
    print(
        "建图基准开始：直行={} mm/s，转向={} mrad/s。".format(
            args.speed,
            args.turn_speed,
        )
    )
    try:
        for index, scan in enumerate(lidar.scans()):
            if index >= args.scans:
                break
            drive_for_scan(client, index, args.speed, args.turn_speed)
            update = slam.process(scan)
            accepted += int(update.accepted)
            integrated += int(update.map_integrated)
            if np.isfinite(update.rmse_m):
                errors.append(float(update.rmse_m))
            scan_samples.append({
                "scan_index": index + 1,
                "timestamp_s": float(scan.timestamp_s),
                "accepted": int(update.accepted),
                "map_integrated": int(update.map_integrated),
                "pose_x_m": float(update.pose.x_m),
                "pose_y_m": float(update.pose.y_m),
                "pose_yaw_deg": float(
                    update.pose.yaw_rad * 180.0 / np.pi
                ),
                "rmse_m": (
                    float(update.rmse_m)
                    if np.isfinite(update.rmse_m)
                    else ""
                ),
                "correspondences": int(update.correspondences),
                "scan_points": int(update.scan_points),
                "inlier_ratio": float(update.inlier_ratio),
                "translation_m": float(update.translation_m),
                "rotation_deg": float(
                    update.rotation_rad * 180.0 / np.pi
                ),
                "linear_speed_m_s": float(update.linear_speed_m_s),
                "angular_speed_rad_s": float(update.angular_speed_rad_s),
                "rejection_reason": update.rejection_reason,
                "map_status": update.map_status,
            })
            print(
                "\rscan={:3d}/{:3d} accepted={:3d} integrated={:3d} "
                "pose=({:+.2f},{:+.2f},{:+.1f}°) rmse={:.3f}m".format(
                    index + 1,
                    args.scans,
                    accepted,
                    integrated,
                    update.pose.x_m,
                    update.pose.y_m,
                    update.pose.yaw_rad * 180.0 / np.pi,
                    update.rmse_m,
                ),
                end="",
                flush=True,
            )
    except SimulatorUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        print()
        try:
            client.stop()
        finally:
            lidar.close()
            client.close(send_cancel=False)

    paths = slam.save(output)
    observed = int(np.count_nonzero(slam.grid.observed))
    occupied = int(
        np.count_nonzero(
            slam.grid.observed & (slam.grid.probabilities() >= 0.65)
        )
    )
    accepted_ratio = accepted / max(1, args.scans)
    mean_rmse = float(np.mean(errors)) if errors else float("inf")
    elapsed = time.monotonic() - started_at
    thresholds = {
        "accepted_ratio_min": 0.55,
        "integrated_scans_min": max(8, args.scans // 12),
        "observed_cells_min": 400,
        "occupied_cells_min": 20,
        "mean_rmse_max_m": 0.16,
    }
    passed = all((
        accepted_ratio >= thresholds["accepted_ratio_min"],
        integrated >= thresholds["integrated_scans_min"],
        observed >= thresholds["observed_cells_min"],
        occupied >= thresholds["occupied_cells_min"],
        mean_rmse <= thresholds["mean_rmse_max_m"],
    ))
    trajectory_length_m = sum(
        float(np.hypot(
            current[1] - previous[1],
            current[2] - previous[2],
        ))
        for previous, current in zip(slam.trajectory, slam.trajectory[1:])
    )
    summary = {
        "passed": passed,
        "requested_scans": args.scans,
        "completed_scans": len(scan_samples),
        "accepted_scans": accepted,
        "rejected_scans": len(scan_samples) - accepted,
        "accepted_ratio": accepted_ratio,
        "integrated_scans": integrated,
        "observed_cells": observed,
        "occupied_cells": occupied,
        "mean_rmse_m": mean_rmse,
        "median_rmse_m": (
            float(np.median(errors)) if errors else float("inf")
        ),
        "p95_rmse_m": (
            float(np.percentile(errors, 95)) if errors else float("inf")
        ),
        "max_rmse_m": max(errors) if errors else float("inf"),
        "elapsed_s": elapsed,
        "scans_per_second": len(scan_samples) / max(elapsed, 1e-9),
        "trajectory_samples": len(slam.trajectory),
        "trajectory_length_m": trajectory_length_m,
        "final_pose": (
            {
                "x_m": scan_samples[-1]["pose_x_m"],
                "y_m": scan_samples[-1]["pose_y_m"],
                "yaw_deg": scan_samples[-1]["pose_yaw_deg"],
            }
            if scan_samples else None
        ),
    }
    report_paths = ()
    if not args.no_report:
        report_paths = generate_mapping_report(
            output,
            parameters={
                "测试类型": "网页仿真建图基准",
                "请求扫描帧数": args.scans,
                "直行速度（mm/s）": args.speed,
                "转向速度（mrad/s）": args.turn_speed,
                "仿真服务地址": args.base_url,
                "地图分辨率（m/格）": slam.grid.resolution_m,
                "地图宽度（格）": slam.grid.width,
                "地图高度（格）": slam.grid.height,
                "输出前缀": str(output),
            },
            summary=summary,
            thresholds=thresholds,
            map_paths=paths,
            scan_samples=scan_samples,
        )
    print("\n=== 建图验收结果 ===")
    print(f"扫描匹配接受率：{accepted_ratio:.1%}（建议 ≥ 55%）")
    print(
        f"地图融合帧数：  {integrated}"
        f"（建议 ≥ {thresholds['integrated_scans_min']}）"
    )
    print(f"已观测栅格数：  {observed}（建议 ≥ 400）")
    print(f"占用栅格数：    {occupied}（建议 ≥ 20）")
    print(f"平均匹配 RMSE： {mean_rmse:.3f} m（建议 ≤ 0.160 m）")
    print(f"耗时：          {elapsed:.1f} s")
    print("地图文件：      {}".format(", ".join(str(path) for path in paths)))
    if report_paths:
        print("分析报告：      {}".format(report_paths[0]))
        print("实验数据：      {}, {}".format(
            report_paths[1],
            report_paths[2],
        ))
    print("结论：          {}".format("PASS，已形成有效地图" if passed else "FAIL，请检查上方指标和地图图像"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
