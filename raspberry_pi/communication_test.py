"""树莓派与 ESP32 的分级通信测试。

默认只做无运动的心跳和状态查询。IO 测试与短距离运动测试必须显式传入参数。
"""

import argparse
import statistics
import time

from car.protocol.messages import LED_BLINK, LED_OFF
from raspberry_pi.config import SerialConfig
from raspberry_pi.esp32 import Esp32Client


def build_argument_parser():
    parser = argparse.ArgumentParser(description="ESP32 Vehicle Link V2 通信测试")
    parser.add_argument("--port", default="/dev/serial0")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument(
        "--link", choices=("auto", "uart", "ble"), default="auto",
        help="auto 优先 UART，超时后切换 BLE",
    )
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--balanced-stress-count",
        type=int,
        default=0,
        help=(
            "send this many zero-speed two-chunk balanced commands; "
            "wheels remain stopped"
        ),
    )
    parser.add_argument("--io-test", action="store_true", help="闪灯并鸣叫一次")
    parser.add_argument(
        "--motion-test",
        action="store_true",
        help="让悬空车轮以 100 mm/s 运行 0.5 秒；运行前必须架空车轮",
    )
    parser.add_argument(
        "--arm-test", action="store_true",
        help="关节0从1500us移动到1550us再复位；运行前确认机械臂无障碍",
    )
    return parser


def print_status(status):
    print(
        "状态：uptime={uptime_ms}ms flags=0x{flags:04X} "
        "v={linear_mm_s}mm/s w={angular_mrad_s}mrad/s "
        "joints={joint_positions} battery={battery_mv}mV errors={bus_errors}".format(
            **status
        )
    )


def run(args):
    config = SerialConfig(
        port=args.port,
        baudrate=args.baud,
        link_mode=args.link,
        ble_device_name=args.ble_name,
        ble_address=args.ble_address,
        wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
    )
    client = Esp32Client(config)
    latencies_ms = []
    try:
        # 测试时逐个同步请求，结果更容易解释，不启动后台心跳线程。
        for index in range(args.count):
            started = time.monotonic()
            client.heartbeat()
            latency = (time.monotonic() - started) * 1000.0
            latencies_ms.append(latency)
            print("心跳 {}/{}：{:.1f} ms".format(index + 1, args.count, latency))
            time.sleep(0.05)

        print_status(client.query_status())
        print(
            "通信通过：链路={}，平均 {:.1f} ms，最大 {:.1f} ms".format(
                client.active_transport,
                statistics.mean(latencies_ms), max(latencies_ms),
            )
        )

        if args.balanced_stress_count > 0:
            stress_latencies_ms = []
            print(
                "开始零速度双分片运动帧压力测试，共 {} 次……".format(
                    args.balanced_stress_count
                )
            )
            for index in range(args.balanced_stress_count):
                started = time.monotonic()
                client.set_twist(0, 0, 450)
                stress_latencies_ms.append(
                    (time.monotonic() - started) * 1000.0
                )
                if (index + 1) % 25 == 0:
                    print(
                        "压力测试 {}/{}".format(
                            index + 1,
                            args.balanced_stress_count,
                        )
                    )
                time.sleep(0.05)
            client.stop()
            print(
                "双分片压力测试通过：平均 {:.1f} ms，最大 {:.1f} ms".format(
                    statistics.mean(stress_latencies_ms),
                    max(stress_latencies_ms),
                )
            )

        if args.io_test:
            print("执行 IO 测试……")
            client.set_led(LED_BLINK, 150)
            client.beep(1, 150, 100)
            time.sleep(0.6)
            client.set_led(LED_OFF)

        if args.motion_test:
            print("执行 0.5 秒低速运动测试……")
            client.start()
            client.set_twist(500, 0, 600)
            time.sleep(2)
            client.stop()
            print("运动测试已停车")

        if args.arm_test:
            print("执行关节0小幅运动测试……")
            client.set_arm_joints([(0, 1700)], duration_ms=500)
            time.sleep(0.8)
            client.set_arm_joints([(0, 1500)], duration_ms=500)
            time.sleep(0.8)
            client.arm_stop()
            print("机械臂测试已复位并停止")
    finally:
        client.close()


def main():
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
