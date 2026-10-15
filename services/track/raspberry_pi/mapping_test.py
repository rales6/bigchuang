"""镭神 N10 独立二维建图测试，不需要连接 ESP32。

示例：
    python3 -m raspberry_pi.mapping_test --port /dev/ttyACM0 \
        --count 200 --output maps/test_room

测试使用相邻激光帧 ICP 估计位姿。采集期间可缓慢、平稳地手推小车；程序没有
编码器里程计和回环检测，因此只适合小范围功能验证。
"""

import argparse
import math

from raspberry_pi.config import LidarConfig, MappingConfig
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import LidarSlam, filter_scan_sector


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="镭神 N10 独立建图测试（无需连接 ESP32）"
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="N10 串口设备")
    parser.add_argument("--count", type=int, default=200, help="最多读取的完整扫描圈数")
    parser.add_argument("--output", default="maps/test_map", help="输出文件前缀")
    parser.add_argument("--resolution", type=float, default=0.05, help="地图分辨率，米/格")
    parser.add_argument("--map-size", type=float, default=20.0, help="正方形地图边长，米")
    parser.add_argument("--min-distance", type=float, default=0.12, help="最小有效距离，米")
    parser.add_argument("--max-distance", type=float, default=8.0, help="最大有效距离，米")
    parser.add_argument("--angle-offset", type=float, default=0.0, help="雷达零度安装偏角")
    parser.add_argument(
        "--field-of-view",
        type=float,
        default=180.0,
        help="保留的有效视场角；默认丢弃受车身影响的后 180°",
    )
    parser.add_argument(
        "--view-center",
        type=float,
        default=0.0,
        help="有效视场中心角，度；0 表示车头方向，正值为车体坐标逆时针",
    )
    parser.add_argument(
        "--counterclockwise",
        action="store_true",
        help="雷达原始角度逆时针递增时使用；N10 默认不需要",
    )
    parser.add_argument(
        "--no-motor-control",
        action="store_true",
        help="不发送 N10 电机启动和停止指令",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每读取 N 圈自动保存；0 表示只在退出时保存",
    )
    return parser


def run(args, lidar=None):
    if args.count <= 0:
        raise ValueError("--count 必须大于 0")
    if args.resolution <= 0 or args.map_size <= 0:
        raise ValueError("--resolution 和 --map-size 必须大于 0")
    if not 0 <= args.min_distance < args.max_distance:
        raise ValueError("距离范围必须满足 0 <= min-distance < max-distance")
    if not 0 < args.field_of_view <= 360:
        raise ValueError("--field-of-view 必须在 0..360 度范围内")

    cells = int(round(args.map_size / args.resolution))
    lidar_config = LidarConfig(
        port=args.port,
        min_distance_m=args.min_distance,
        max_distance_m=args.max_distance,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        motor_control=not args.no_motor_control,
    )
    slam = LidarSlam(
        MappingConfig(
            resolution_m=args.resolution,
            width_cells=cells,
            height_cells=cells,
        ),
        lidar_config,
    )
    lidar = lidar or N10LidarDriver(lidar_config)
    read_count = 0
    accepted_count = 0

    print("独立建图测试已启动；可缓慢手推小车，按 Ctrl+C 提前保存并退出。")
    try:
        for scan in lidar.scans():
            read_count += 1
            scan = filter_scan_view(
                scan,
                field_of_view_deg=args.field_of_view,
                center_deg=args.view_center,
            )
            update = slam.process(scan)
            if update.accepted:
                accepted_count += 1
            rmse = "{:.3f}".format(update.rmse_m) if math.isfinite(update.rmse_m) else "inf"
            print(
                "\rscan={}/{} accepted={} pose=({:+.2f}, {:+.2f}, {:+.1f}deg) "
                "rmse={}m points={}    ".format(
                    read_count,
                    args.count,
                    accepted_count,
                    update.pose.x_m,
                    update.pose.y_m,
                    math.degrees(update.pose.yaw_rad),
                    rmse,
                    update.scan_points,
                ),
                end="",
                flush=True,
            )
            if args.save_every > 0 and read_count % args.save_every == 0:
                slam.save(args.output)
            if read_count >= args.count:
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止采集。")
    finally:
        lidar.close()

    paths = slam.save(args.output)
    print(
        "\n建图结束：读取 {} 圈，接受 {} 圈；已保存：{}".format(
            read_count,
            accepted_count,
            ", ".join(str(path) for path in paths),
        )
    )
    return paths


def filter_scan_view(scan, field_of_view_deg=360.0, center_deg=0.0):
    """兼容旧测试调用，实际过滤统一由 mapping.scan_filter 实现。"""
    return filter_scan_sector(
        scan,
        center_deg=center_deg,
        field_of_view_deg=field_of_view_deg,
    )


def main():
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
