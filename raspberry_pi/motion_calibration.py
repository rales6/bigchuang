"""用雷达短程定位标定小车直行和右转参数。

这个工具不会写入正式地图。每次试验都会创建一个临时 SLAM 实例，用运动前、
运动中和停车后的扫描估计实际位移及转角，并把结果保存成 CSV。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import math
from pathlib import Path
import statistics
import time

from raspberry_pi.config import LidarConfig, MappingConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import LidarSlam, filter_scan_sector


@dataclass(frozen=True)
class CalibrationResult:
    trial: int
    mode: str
    command_linear_mm_s: int
    command_angular_mrad_s: int
    duration_s: float
    x_m: float
    y_m: float
    distance_m: float
    yaw_deg: float
    accepted_scans: int
    rejected_scans: int
    mean_rmse_m: float
    mean_inlier_ratio: float


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="使用 N10 雷达测量小车直行偏差和右转实际转角",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="必须显式给出；否则只检查参数，不会驱动车轮",
    )
    parser.add_argument(
        "--mode",
        choices=("straight", "right", "all", "sweep"),
        default="straight",
        help="测试直行、右转、两者，或交互式多速度扫描拟合",
    )
    parser.add_argument(
        "--sweep-start",
        type=int,
        default=350,
        help="sweep 模式的起始直行速度，单位 mm/s",
    )
    parser.add_argument(
        "--sweep-step",
        type=int,
        default=25,
        help="每完成一档后增加的速度，单位 mm/s",
    )
    parser.add_argument(
        "--sweep-max",
        type=int,
        default=550,
        help="速度扫描安全上限，单位 mm/s",
    )
    parser.add_argument(
        "--fit-strength",
        type=float,
        default=0.50,
        help="本轮拟合补偿的应用比例；默认只应用 50%% 防止过调",
    )
    parser.add_argument(
        "--linear-speeds",
        default="250,300,350,400",
        help="直行速度列表，单位 mm/s",
    )
    parser.add_argument(
        "--turn-speeds",
        default="1500,2000,2500,3000",
        help="右转角速度绝对值列表，单位 mrad/s",
    )
    parser.add_argument("--duration", type=float, default=0.60)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.35,
        help="停车后继续采集稳定扫描的时间",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过每次试验前的人工确认；首次实车测试不建议使用",
    )
    parser.add_argument("--command-ttl-ms", type=int, default=600)
    parser.add_argument("--link", choices=("uart", "ble", "auto"), default="ble")
    parser.add_argument("--esp-port", default="/dev/serial0")
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default=None)
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument("--counterclockwise", action="store_true")
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--front-center", type=float, default=0.0)
    parser.add_argument("--usable-fov", type=float, default=160.0)
    parser.add_argument(
        "--track-width-mm",
        type=float,
        default=185.0,
        help="用于计算左右轮相对补偿；sweep模式经确认后可写入ESP32",
    )
    parser.add_argument(
        "--output",
        default="calibration/motion_calibration.csv",
    )
    return parser


def parse_positive_ints(value, name):
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("{} 必须是逗号分隔的整数".format(name)) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("{} 必须包含正整数".format(name))
    return parsed


def validate_args(args):
    linear_speeds = parse_positive_ints(args.linear_speeds, "--linear-speeds")
    turn_speeds = parse_positive_ints(args.turn_speeds, "--turn-speeds")
    errors = []
    if any(speed > 550 for speed in linear_speeds):
        errors.append("直行速度不能超过 550 mm/s")
    if any(speed > 3500 for speed in turn_speeds):
        errors.append("转向速度不能超过 3500 mrad/s")
    if not 0.15 <= args.duration <= 3.0:
        errors.append("--duration 必须在 0.15..3.0 秒之间")
    if not 1 <= args.repeat <= 10:
        errors.append("--repeat 必须在 1..10 之间")
    if not 0.1 <= args.settle_time <= 2.0:
        errors.append("--settle-time 必须在 0.1..2.0 秒之间")
    if not 200 <= args.command_ttl_ms <= 700:
        errors.append("--command-ttl-ms 必须在 200..700 ms 之间")
    if not 30.0 <= args.usable_fov <= 180.0:
        errors.append("--usable-fov 必须在 30..180 度之间")
    if not 100.0 <= args.track_width_mm <= 400.0:
        errors.append("--track-width-mm 必须在 100..400 mm 之间")
    if not 50 <= args.sweep_start <= 550:
        errors.append("--sweep-start 必须在 50..550 mm/s 之间")
    if not 5 <= args.sweep_step <= 200:
        errors.append("--sweep-step 必须在 5..200 mm/s 之间")
    if not args.sweep_start <= args.sweep_max <= 550:
        errors.append("--sweep-max 必须在 sweep-start..550 之间")
    if not 0.10 <= args.fit_strength <= 1.0:
        errors.append("--fit-strength 必须在 0.10..1.0 之间")
    if errors:
        raise ValueError("；".join(errors))
    return linear_speeds, turn_speeds


def build_trials(mode, linear_speeds, turn_speeds, repeat):
    commands = []
    if mode in ("straight", "all"):
        commands.extend(("straight", speed, 0) for speed in linear_speeds)
    if mode in ("right", "all"):
        # 本项目约定角速度为负表示右转。
        commands.extend(("right", 0, -speed) for speed in turn_speeds)
    return [
        (trial, motion, linear, angular)
        for motion, linear, angular in commands
        for trial in range(1, repeat + 1)
    ]


def sweep_speeds(start, step, maximum):
    return list(range(int(start), int(maximum) + 1, int(step)))


def straight_scale_suggestion(result, track_width_mm):
    """根据差速模型给出下一轮左右输出倍率，不直接改固件。"""
    if result.mode != "straight" or result.distance_m < 0.05:
        return None
    track_width_m = track_width_mm / 1000.0
    forward_m = max(0.001, result.x_m)
    yaw_rad = math.radians(result.yaw_deg)
    left_distance = forward_m - yaw_rad * track_width_m / 2.0
    right_distance = forward_m + yaw_rad * track_width_m / 2.0
    if left_distance <= 0.01 or right_distance <= 0.01:
        return None
    left = forward_m / left_distance
    right = forward_m / right_distance
    normalizer = (left + right) / 2.0
    return left / normalizer, right / normalizer


def turn_gain_suggestion(result):
    if result.mode != "right" or abs(result.yaw_deg) < 3.0:
        return None
    expected_deg = math.degrees(
        abs(result.command_angular_mrad_s) / 1000.0 * result.duration_s
    )
    return expected_deg / abs(result.yaw_deg)


def _calibration_mapping_config():
    # 标定允许的单帧运动范围比正式建图宽，但仍保留 ICP 质量检查。
    return MappingConfig(
        max_match_rmse_m=0.20,
        min_match_inlier_ratio=0.25,
        coarse_yaw_range_deg=55.0,
        coarse_yaw_step_deg=2.0,
        max_pose_linear_speed_m_s=1.20,
        max_pose_translation_step_m=0.35,
        max_pose_angular_speed_rad_s=10.0,
        max_pose_rotation_step_deg=45.0,
        map_min_linear_speed_m_s=0.0,
        map_max_linear_speed_m_s=2.0,
        map_max_angular_speed_rad_s=12.0,
    )


def _next_usable_scan(scan_iterator, args):
    while True:
        scan = filter_scan_sector(
            next(scan_iterator),
            args.front_center,
            args.usable_fov,
        )
        if len(scan.points()) >= 45:
            return scan


def run_trial(
    client,
    scan_iterator,
    args,
    trial,
    mode,
    linear_mm_s,
    angular_mrad_s,
):
    slam = LidarSlam(
        _calibration_mapping_config(),
        LidarConfig(
            min_scan_points=45,
            angle_offset_deg=args.angle_offset,
            clockwise=not args.counterclockwise,
        ),
    )
    initial = _next_usable_scan(scan_iterator, args)
    slam.process(initial)
    accepted = 1
    rejected = 0
    rmses = []
    inliers = []

    started = time.monotonic()
    actual_duration_s = 0.0
    try:
        client.set_twist(
            linear_mm_s,
            angular_mrad_s,
            ttl_ms=args.command_ttl_ms,
        )
        while time.monotonic() - started < args.duration:
            update = slam.process(_next_usable_scan(scan_iterator, args))
            if update.accepted:
                accepted += 1
                if math.isfinite(update.rmse_m):
                    rmses.append(update.rmse_m)
                inliers.append(update.inlier_ratio)
            else:
                rejected += 1
    finally:
        actual_duration_s = time.monotonic() - started
        client.stop()

    settle_deadline = time.monotonic() + args.settle_time
    while time.monotonic() < settle_deadline:
        update = slam.process(_next_usable_scan(scan_iterator, args))
        if update.accepted:
            accepted += 1
            if math.isfinite(update.rmse_m):
                rmses.append(update.rmse_m)
            inliers.append(update.inlier_ratio)
        else:
            rejected += 1

    pose = slam.pose
    return CalibrationResult(
        trial=trial,
        mode=mode,
        command_linear_mm_s=linear_mm_s,
        command_angular_mrad_s=angular_mrad_s,
        duration_s=actual_duration_s,
        x_m=pose.x_m,
        y_m=pose.y_m,
        distance_m=math.hypot(pose.x_m, pose.y_m),
        yaw_deg=math.degrees(pose.yaw_rad),
        accepted_scans=accepted,
        rejected_scans=rejected,
        mean_rmse_m=statistics.mean(rmses) if rmses else math.inf,
        mean_inlier_ratio=statistics.mean(inliers) if inliers else 0.0,
    )


def save_results(path, results):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field_names = [item.name for item in fields(CalibrationResult)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for result in results:
            writer.writerow({
                name: getattr(result, name)
                for name in field_names
            })
    return output


def _print_result(result, track_width_mm):
    print(
        "结果：位移=({:+.3f},{:+.3f})m 距离={:.3f}m "
        "偏航={:+.1f}° 接受/拒绝={}/{} RMSE={:.3f}m".format(
            result.x_m,
            result.y_m,
            result.distance_m,
            result.yaw_deg,
            result.accepted_scans,
            result.rejected_scans,
            result.mean_rmse_m,
        )
    )
    scales = straight_scale_suggestion(result, track_width_mm)
    if scales is not None:
        print(
            "  下一轮左右输出倍率建议：LEFT={:.3f}, RIGHT={:.3f}"
            "（先记录，完成多次测试后取中位数）".format(*scales)
        )
    gain = turn_gain_suggestion(result)
    if gain is not None:
        print(
            "  角速度增益倍率建议：当前 ANGULAR_OUTPUT_GAIN × {:.3f}"
            "（短脉冲含加速影响，仅作初值）".format(gain)
        )


def summarize_results(results, track_width_mm):
    """按相同命令汇总重复测试，以中位数抑制单次 ICP 异常。"""
    groups = {}
    for result in results:
        key = (
            result.mode,
            result.command_linear_mm_s,
            result.command_angular_mrad_s,
        )
        groups.setdefault(key, []).append(result)

    summaries = []
    for key, group in groups.items():
        mode, linear, angular = key
        summary = {
            "mode": mode,
            "linear_mm_s": linear,
            "angular_mrad_s": angular,
            "samples": len(group),
            "x_m": statistics.median(item.x_m for item in group),
            "y_m": statistics.median(item.y_m for item in group),
            "yaw_deg": statistics.median(item.yaw_deg for item in group),
        }
        if mode == "straight":
            scale_pairs = [
                suggestion
                for suggestion in (
                    straight_scale_suggestion(item, track_width_mm)
                    for item in group
                )
                if suggestion is not None
            ]
            if scale_pairs:
                summary["left_scale"] = statistics.median(
                    pair[0] for pair in scale_pairs
                )
                summary["right_scale"] = statistics.median(
                    pair[1] for pair in scale_pairs
                )
        else:
            gains = [
                suggestion
                for suggestion in (
                    turn_gain_suggestion(item) for item in group
                )
                if suggestion is not None
            ]
            if gains:
                summary["angular_gain_multiplier"] = statistics.median(gains)
        summaries.append(summary)
    return summaries


def fit_speed_dependent_trim(results, track_width_mm):
    """拟合 ``trim = intercept + slope * speed``。

    每个速度先对重复测试取中位数，再做最小二乘，避免某一帧直接支配拟合。
    正 trim 表示降低左侧并提高右侧。
    """
    points = []
    for summary in summarize_results(results, track_width_mm):
        if summary["mode"] != "straight" or "left_scale" not in summary:
            continue
        speed = float(summary["linear_mm_s"])
        trim = (
            float(summary["right_scale"])
            - float(summary["left_scale"])
        ) / 2.0
        points.append((speed, trim))
    points.sort()
    if len(points) < 2 or len({point[0] for point in points}) < 2:
        raise ValueError("至少需要两个不同速度的有效数据才能拟合")

    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _y in points)
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    ) / denominator
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x, _y in points]
    residual = sum(
        (y - estimate) ** 2
        for (_x, y), estimate in zip(points, predicted)
    )
    total = sum((y - mean_y) ** 2 for _x, y in points)
    r_squared = 1.0 if total <= 1e-12 else 1.0 - residual / total

    # ESP32运行时仍会限幅；这里先把整段 0..550 的最大补偿压到12%，
    # 留出协议硬限制15%的安全余量。
    endpoint_peak = max(abs(intercept), abs(intercept + slope * 550.0))
    if endpoint_peak > 0.12:
        factor = 0.12 / endpoint_peak
        intercept *= factor
        slope *= factor
    return {
        "trim_intercept": intercept,
        "trim_slope_per_mm_s": slope,
        "r_squared": r_squared,
        "points": points,
    }


def combine_calibration(current, fitted, strength):
    combined = {
        "trim_intercept": (
            float(current["trim_intercept"])
            + float(fitted["trim_intercept"]) * float(strength)
        ),
        "trim_slope_per_mm_s": (
            float(current["trim_slope_per_mm_s"])
            + float(fitted["trim_slope_per_mm_s"]) * float(strength)
        ),
    }
    peak = max(
        abs(combined["trim_intercept"]),
        abs(
            combined["trim_intercept"]
            + combined["trim_slope_per_mm_s"] * 550.0
        ),
    )
    if peak > 0.14:
        factor = 0.14 / peak
        combined["trim_intercept"] *= factor
        combined["trim_slope_per_mm_s"] *= factor
    return combined


def _print_summary(results, track_width_mm):
    print("\n重复测试中位数汇总：")
    for summary in summarize_results(results, track_width_mm):
        command = (
            "直行 {} mm/s".format(summary["linear_mm_s"])
            if summary["mode"] == "straight"
            else "右转 {} mrad/s".format(abs(summary["angular_mrad_s"]))
        )
        line = (
            "{}：x={:+.3f}m y={:+.3f}m yaw={:+.1f}° (n={})".format(
                command,
                summary["x_m"],
                summary["y_m"],
                summary["yaw_deg"],
                summary["samples"],
            )
        )
        if "left_scale" in summary:
            line += " 建议 LEFT={:.3f} RIGHT={:.3f}".format(
                summary["left_scale"],
                summary["right_scale"],
            )
        if "angular_gain_multiplier" in summary:
            line += " 建议角速度增益×{:.3f}".format(
                summary["angular_gain_multiplier"]
            )
        print(line)


def _run_sweep(client, scan_iterator, args):
    results = []
    finish_requested = False
    speeds = sweep_speeds(
        args.sweep_start,
        args.sweep_step,
        args.sweep_max,
    )
    for speed_index, speed in enumerate(speeds, 1):
        while True:
            group_start = len(results)
            print(
                "\n===== 速度档 {}/{}：{} mm/s =====".format(
                    speed_index, len(speeds), speed
                )
            )
            for trial in range(1, args.repeat + 1):
                if not args.yes:
                    input(
                        "摆正/复位小车并确认周围安全，按 Enter 开始"
                        "第 {}/{} 次：".format(trial, args.repeat)
                    )
                result = run_trial(
                    client,
                    scan_iterator,
                    args,
                    trial,
                    "straight",
                    speed,
                    0,
                )
                results.append(result)
                _print_result(result, args.track_width_mm)

            group = results[group_start:]
            _print_summary(group, args.track_width_mm)
            if args.yes:
                action = "f" if speed_index == len(speeds) else "c"
            else:
                action = input(
                    "选择：[c]保留并测试下一档  [f]保留并停止、立即拟合  "
                    "[d]舍弃本档并重测  [q]停止且不写入ESP32："
                ).strip().lower() or "c"
            if action == "d":
                del results[group_start:]
                print("已舍弃 {} mm/s 的本组数据，重新测试该速度。".format(speed))
                continue
            if action == "q":
                return results, None, False
            if action == "f":
                try:
                    fit_speed_dependent_trim(
                        results, args.track_width_mm
                    )
                    finish_requested = True
                except ValueError as exc:
                    if speed_index < len(speeds):
                        print("{}；继续测试下一档。".format(exc))
                    else:
                        print("{}；无法写入ESP32。".format(exc))
                        return results, None, False
            elif action != "c":
                print("无法识别的选择，按保留并继续处理。")
            break
        if finish_requested:
            break

    try:
        fitted = fit_speed_dependent_trim(results, args.track_width_mm)
    except ValueError as exc:
        print("拟合失败：{}；未写入ESP32。".format(exc))
        return results, None, False
    print("\n速度相关左右轮补偿拟合：")
    for speed, trim in fitted["points"]:
        print("  speed={:.0f} mm/s -> trim={:+.4f}".format(speed, trim))
    print(
        "  trim(speed) = {:+.6f} {:+.9f} × speed，R²={:.3f}".format(
            fitted["trim_intercept"],
            fitted["trim_slope_per_mm_s"],
            fitted["r_squared"],
        )
    )
    return results, fitted, True


def _query_current_calibration(client):
    try:
        current = client.query_drive_calibration()
        print(
            "ESP32当前标定：intercept={:+.6f}, slope={:+.9f}".format(
                current["trim_intercept"],
                current["trim_slope_per_mm_s"],
            )
        )
        return current, True
    except Exception as exc:
        print(
            "ESP32尚不支持蓝牙标定参数查询：{}；"
            "本次仍可测试和拟合，但不能自动写入。".format(exc)
        )
        return {
            "trim_intercept": 0.0,
            "trim_slope_per_mm_s": 0.0,
        }, False


def _confirm_and_write_calibration(client, args, current, fitted, can_write):
    candidate = combine_calibration(current, fitted, args.fit_strength)
    print(
        "\n准备写入的最终参数（本轮应用 {:.0f}%）："
        "intercept={:+.6f}, slope={:+.9f}".format(
            args.fit_strength * 100.0,
            candidate["trim_intercept"],
            candidate["trim_slope_per_mm_s"],
        )
    )
    for speed in (
        args.sweep_start,
        min(args.sweep_max, 550),
    ):
        trim = (
            candidate["trim_intercept"]
            + candidate["trim_slope_per_mm_s"] * speed
        )
        print(
            "  {} mm/s：LEFT={:.3f}, RIGHT={:.3f}".format(
                speed, 1.0 - trim, 1.0 + trim
            )
        )
    if not can_write:
        print("未写入：请先通过有线方式更新一次ESP32端 car/ 固件。")
        return False
    answer = input(
        "确认通过当前BLE/UART链路写入ESP32并掉电保存？[y/N]："
    ).strip().lower()
    if answer not in ("y", "yes"):
        print("用户取消写入，拟合结果仍保存在CSV中。")
        return False
    client.stop()
    client.set_drive_calibration(
        candidate["trim_intercept"],
        candidate["trim_slope_per_mm_s"],
    )
    verified = client.query_drive_calibration()
    if (
        abs(
            verified["trim_intercept"]
            - candidate["trim_intercept"]
        ) > 1e-5
        or abs(
            verified["trim_slope_per_mm_s"]
            - candidate["trim_slope_per_mm_s"]
        ) > 1e-7
    ):
        raise RuntimeError("ESP32回读的标定参数与写入值不一致")
    print("ESP32标定参数写入并回读验证成功。")
    return True


def run(args, client=None, lidar=None):
    linear_speeds, turn_speeds = validate_args(args)
    trials = (
        [
            (trial, "straight", speed, 0)
            for speed in sweep_speeds(
                args.sweep_start,
                args.sweep_step,
                args.sweep_max,
            )
            for trial in range(1, args.repeat + 1)
        ]
        if args.mode == "sweep"
        else build_trials(
            args.mode,
            linear_speeds,
            turn_speeds,
            args.repeat,
        )
    )
    if not args.enable_motion:
        print("参数检查通过；未给出 --enable-motion，车轮不会运动。")
        print("计划执行 {} 次试验。".format(len(trials)))
        return []

    lidar_config = LidarConfig(
        port=args.lidar_port,
        baudrate=args.lidar_baud,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        min_scan_points=45,
    )
    client = client or Esp32Client(SerialConfig(
        port=args.esp_port,
        link_mode=args.link,
        ble_device_name=args.ble_name,
        ble_address=args.ble_address,
    ))
    lidar = lidar or N10LidarDriver(lidar_config)
    results = []
    try:
        client.heartbeat()
        client.start()
        print("通信通过：当前链路={}".format(client.active_transport))
        print("请清空测试区域，并随时准备按 Ctrl+C；任何异常都会发送停车命令。")
        scan_iterator = iter(lidar.scans())
        if args.mode == "sweep":
            current, can_write = _query_current_calibration(client)
            results, fitted, allow_write = _run_sweep(
                client, scan_iterator, args
            )
            if fitted is not None and allow_write:
                _confirm_and_write_calibration(
                    client,
                    args,
                    current,
                    fitted,
                    can_write,
                )
        else:
            for index, (trial, mode, linear, angular) in enumerate(trials, 1):
                command = (
                    "直行 {} mm/s".format(linear)
                    if mode == "straight"
                    else "右转 {} mrad/s".format(abs(angular))
                )
                print(
                    "\n试验 {}/{}：{}，持续 {:.2f}s，第 {} 次".format(
                        index, len(trials), command, args.duration, trial
                    )
                )
                if not args.yes:
                    input(
                        "摆正/复位小车并确认周围安全，然后按 Enter 开始；"
                        "Ctrl+C 取消："
                    )
                result = run_trial(
                    client,
                    scan_iterator,
                    args,
                    trial,
                    mode,
                    linear,
                    angular,
                )
                results.append(result)
                _print_result(result, args.track_width_mm)
    except KeyboardInterrupt:
        print("\n用户取消测试。")
    finally:
        try:
            client.stop()
        except Exception as exc:
            print("停车命令告警（命令 TTL 到期后仍会失效）：{}".format(exc))
        lidar.close()
        client.close()

    if results:
        output = save_results(args.output, results)
        _print_summary(results, args.track_width_mm)
        print("\n已保存 {} 条结果：{}".format(len(results), output))
        print("请以三次重复测试的中位数为准，不要根据单次结果直接修改固件。")
    return results


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
