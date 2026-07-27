"""树莓派激光雷达建图入口。

示例：
    python3 -m raspberry_pi.mapping_app --esp-port /dev/serial0 \
        --lidar-port /dev/ttyUSB0 --output maps/room
"""

import argparse
import signal
import time

from raspberry_pi.config import LidarConfig, MappingConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import LidarSlam


def build_argument_parser():
    parser = argparse.ArgumentParser(description="镭神 N10 + ESP32 二维栅格建图")
    parser.add_argument("--esp-port", default="/dev/serial0", help="ESP32 UART 设备")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0", help="雷达串口设备")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument(
        "--counterclockwise",
        action="store_true",
        help="仅在雷达角度实际逆时针递增时使用；N10 默认不需要",
    )
    parser.add_argument("--angle-offset", type=float, default=0.0, help="雷达零度安装偏角")
    parser.add_argument(
        "--no-motor-control",
        action="store_true",
        help="不发送 N10 电机启动/停止指令",
    )
    parser.add_argument("--resolution", type=float, default=0.05, help="地图分辨率，米/格")
    parser.add_argument("--map-size", type=float, default=20.0, help="正方形地图边长，米")
    parser.add_argument("--output", default="maps/map", help="输出文件前缀")
    parser.add_argument("--save-every", type=int, default=50, help="每 N 个有效扫描保存一次")
    parser.add_argument("--max-scans", type=int, default=0, help="0 表示持续运行")
    return parser


def run(args):
    cells = int(round(args.map_size / args.resolution))
    lidar_config = LidarConfig(
        port=args.lidar_port,
        baudrate=args.lidar_baud,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        motor_control=not args.no_motor_control,
    )
    mapping_config = MappingConfig(
        resolution_m=args.resolution,
        width_cells=cells,
        height_cells=cells,
    )
    slam = LidarSlam(mapping_config, lidar_config)
    client = Esp32Client(SerialConfig(port=args.esp_port))
    lidar = N10LidarDriver(lidar_config)
    stopping = False

    def request_stop(_signal_number, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    accepted_count = 0
    client.start()
    print("建图已启动；Ctrl+C 会安全停车并保存地图。")
    try:
        for scan in lidar.scans():
            if stopping:
                break
            update = slam.process(scan)
            if update.accepted:
                accepted_count += 1
            print(
                "\rscan={} accepted={} pose=({:+.2f}, {:+.2f}, {:+.1f}deg) "
                "rmse={:.3f}m points={}    ".format(
                    accepted_count,
                    update.accepted,
                    update.pose.x_m,
                    update.pose.y_m,
                    update.pose.yaw_rad * 180.0 / 3.141592653589793,
                    update.rmse_m,
                    update.scan_points,
                ),
                end="",
                flush=True,
            )
            if update.accepted and args.save_every > 0 and accepted_count % args.save_every == 0:
                slam.save(args.output)
            if args.max_scans > 0 and accepted_count >= args.max_scans:
                break
            if client.last_link_error is not None:
                print("\nESP32 链路告警：{}".format(client.last_link_error))
                time.sleep(0.1)
    finally:
        print("\n正在停车并保存地图……")
        lidar.close()
        client.close()
        paths = slam.save(args.output)
        print("已保存：{}".format(", ".join(str(path) for path in paths)))


def main():
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
