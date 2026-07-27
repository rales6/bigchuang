"""模拟树莓派的交互式 ESP32 指令测试器。

用法：
    python tools/pi_command_console.py COM5
    python3 tools/pi_command_console.py /dev/serial0

依赖：python -m pip install pyserial
"""

import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAR_ROOT = os.path.join(PROJECT_ROOT, "car")
sys.path.insert(0, CAR_ROOT)

from protocol.frame import (  # noqa: E402
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ACK_REQUIRED,
    FLAG_ERROR,
    FrameParser,
    encode_frame,
)
from protocol.messages import (  # noqa: E402
    CANCEL_ALL,
    CANCEL_ARM,
    CANCEL_BUZZER,
    CANCEL_DRIVE,
    CANCEL_LED,
    LED_BLINK,
    LED_OFF,
    LED_ON,
    MSG_ARM_STOP,
    MSG_BEEP,
    MSG_CANCEL,
    MSG_ERROR,
    MSG_HEARTBEAT,
    MSG_QUERY_STATUS,
    MSG_ROBOT_STATUS,
    MSG_SET_ARM_JOINTS,
    MSG_SET_LED,
    MSG_SET_TWIST,
    decode_robot_status,
    encode_arm_joints,
    encode_beep,
    encode_led,
    encode_twist,
)


class RobotClient:
    def __init__(self, port, baudrate=230400):
        try:
            import serial
        except ImportError:
            raise SystemExit("缺少 pyserial，请运行: python -m pip install pyserial")
        self.serial = serial.Serial(port, baudrate, timeout=0.03)
        self.parser = FrameParser()
        self._seq = 0
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = True
        self._active_drive = None
        self.last_status = None
        self._maintenance = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._maintenance.start()

    def request(self, msg_type, payload=b"", timeout=0.45, retries=2):
        with self._request_lock:
            seq = self._next_seq()
            packet = encode_frame(
                msg_type, seq, payload,
                src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32,
                flags=FLAG_ACK_REQUIRED,
            )
            for attempt in range(retries + 1):
                self.serial.write(packet)
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    count = self.serial.in_waiting
                    data = self.serial.read(count or 1)
                    for frame in self.parser.feed(data):
                        if frame.msg_type == MSG_ROBOT_STATUS:
                            self.last_status = decode_robot_status(frame.payload)
                            if frame.seq != seq:
                                continue
                        if frame.seq != seq or not frame.is_response:
                            continue
                        if frame.msg_type == MSG_ERROR or frame.flags & FLAG_ERROR:
                            code = frame.payload[1] if len(frame.payload) >= 2 else -1
                            raise RuntimeError("ESP32 拒绝指令，错误码 {}".format(code))
                        return frame
                if attempt == retries:
                    raise TimeoutError("ESP32 未应答 seq={}".format(seq))

    def set_drive(self, linear, angular, ttl_ms=600):
        with self._state_lock:
            self._active_drive = (linear, angular, ttl_ms)
        self.request(MSG_SET_TWIST, encode_twist(linear, angular, ttl_ms))

    def stop_drive(self):
        with self._state_lock:
            self._active_drive = None
        self.cancel(CANCEL_DRIVE)

    def set_arm(self, joints, duration_ms=800):
        payload = encode_arm_joints(joints, duration_ms)
        self.request(MSG_SET_ARM_JOINTS, payload)

    def set_led(self, mode, period_ms=0):
        self.request(MSG_SET_LED, encode_led(mode, period_ms))

    def beep(self, repeat, on_ms, off_ms):
        self.request(MSG_BEEP, encode_beep(repeat, on_ms, off_ms))

    def cancel(self, mask=CANCEL_ALL):
        if mask & CANCEL_DRIVE:
            with self._state_lock:
                self._active_drive = None
        self.request(MSG_CANCEL, bytes((mask,)))

    def query_status(self):
        frame = self.request(MSG_QUERY_STATUS)
        self.last_status = decode_robot_status(frame.payload)
        return self.last_status

    def close(self):
        self._running = False
        try:
            self.cancel(CANCEL_ALL)
        except Exception:
            pass
        self._maintenance.join(timeout=1.0)
        self.serial.close()

    def _maintenance_loop(self):
        """持续心跳，并刷新有时效的速度命令。"""
        while self._running:
            try:
                with self._state_lock:
                    drive = self._active_drive
                if drive:
                    self.request(MSG_SET_TWIST, encode_twist(*drive), timeout=0.3, retries=1)
                else:
                    self.request(MSG_HEARTBEAT, timeout=0.3, retries=1)
            except Exception as exc:
                print("\n[链路告警]", exc)
            time.sleep(0.25)

    def _next_seq(self):
        value = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        return value


class SequenceRunner:
    """后台执行演示序列；主线程仍可接收 cancel 或新指令。"""

    def __init__(self, client):
        self.client = client
        self._cancel = threading.Event()
        self._thread = None

    def start(self, name, steps):
        self.cancel(send_to_robot=True)
        # 即使之前不是演示序列，也可能仍有手动底盘/机械臂动作在运行。
        self.client.cancel(CANCEL_ALL)
        self._cancel.clear()

        def run():
            print("[序列开始]", name)
            try:
                for delay_s, action in steps:
                    if self._cancel.wait(delay_s):
                        return
                    action()
            except Exception as exc:
                print("[序列失败]", exc)
            finally:
                if not self._cancel.is_set():
                    print("[序列完成]", name)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def cancel(self, send_to_robot=False):
        was_active = self._thread is not None and self._thread.is_alive()
        self._cancel.set()
        if send_to_robot and was_active:
            try:
                self.client.cancel(CANCEL_ALL)
            except Exception:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self._thread = None


class CommandConsole:
    HOME = [(0, 1500), (1, 1700), (2, 2000), (3, 1100), (4, 1500), (5, 1200)]

    def __init__(self, client):
        self.client = client
        self.sequence = SequenceRunner(client)

    def execute(self, line):
        parts = line.strip().lower().split()
        if not parts:
            return True
        name = parts[0]
        args = parts[1:]

        # 演示序列在后台运行；任意新的动作指令都会先中止旧序列和旧动作。
        passive = ("help", "ping", "status", "quit", "exit", "cancel", "stop_all")
        demos = ("demo_drive", "demo_arm", "demo_io", "demo_all")
        if name not in passive and name not in demos:
            self.sequence.cancel(send_to_robot=True)

        if name in ("quit", "exit"):
            return False
        if name == "help":
            print("完整指令见 docs/command_set.md")
        elif name == "ping":
            self.client.request(MSG_HEARTBEAT)
            print("pong")
        elif name == "status":
            print_status(self.client.query_status())
        elif name in ("cancel", "stop_all"):
            self.sequence.cancel()
            self.client.cancel(CANCEL_ALL)
            print("全部动作已取消")
        elif name == "stop_drive":
            self.client.stop_drive()
        elif name == "arm_stop":
            self.client.request(MSG_ARM_STOP)
        elif name == "forward_slow":
            self._drive(180, 0)
        elif name == "forward":
            self._drive(350, 0)
        elif name == "forward_fast":
            self._drive(500, 0)
        elif name == "backward":
            self._drive(-300, 0)
        elif name == "spin_left":
            self._drive(0, 2200)
        elif name == "spin_right":
            self._drive(0, -2200)
        elif name == "curve_left":
            self._drive(300, 900)
        elif name == "curve_right":
            self._drive(300, -900)
        elif name == "drive":
            if len(args) not in (2, 3):
                raise ValueError("用法: drive <linear_mm_s> <angular_mrad_s> [ttl_ms]")
            self._drive(int(args[0]), int(args[1]), int(args[2]) if len(args) == 3 else 600)
        elif name == "arm_home":
            self._arm(self.HOME, 1200)
        elif name == "arm_left":
            self._arm([(0, 2200)], 800)
        elif name == "arm_right":
            self._arm([(0, 800)], 800)
        elif name == "arm_up":
            self._arm([(1, 1600), (2, 1700), (3, 1350)], 1000)
        elif name == "arm_down":
            self._arm([(1, 1200), (2, 2100), (3, 1000)], 1000)
        elif name == "grip":
            self._arm([(5, 1500)], 500)
        elif name == "release":
            self._arm([(5, 1200)], 500)
        elif name == "arm":
            if len(args) not in (2, 3):
                raise ValueError("用法: arm <joint_id> <pulse_us> [duration_ms]")
            self._arm([(int(args[0]), int(args[1]))], int(args[2]) if len(args) == 3 else 800)
        elif name == "led_off":
            self.client.set_led(LED_OFF)
        elif name == "led_on":
            self.client.set_led(LED_ON)
        elif name == "led_blink_slow":
            self.client.set_led(LED_BLINK, 500)
        elif name == "led_blink_fast":
            self.client.set_led(LED_BLINK, 100)
        elif name == "beep_short":
            self.client.beep(1, 100, 100)
        elif name == "beep_3":
            self.client.beep(3, 120, 120)
        elif name == "beep_long":
            self.client.beep(1, 1000, 100)
        elif name == "demo_drive":
            self.sequence.start(name, self._demo_drive())
        elif name == "demo_arm":
            self.sequence.start(name, self._demo_arm())
        elif name == "demo_io":
            self.sequence.start(name, self._demo_io())
        elif name == "demo_all":
            self.sequence.start(name, self._demo_all())
        else:
            raise ValueError("未知指令: {}，输入 help 查看说明".format(name))
        return True

    def _drive(self, linear, angular, ttl=600):
        self.sequence.cancel()
        self.client.cancel(CANCEL_DRIVE)
        self.client.set_drive(linear, angular, ttl)

    def _arm(self, joints, duration):
        self.sequence.cancel()
        self.client.cancel(CANCEL_ARM)
        self.client.set_arm(joints, duration)

    def _demo_drive(self):
        return [
            (0.0, lambda: self.client.set_drive(250, 0)),
            (1.5, lambda: self.client.set_drive(0, 1800)),
            (1.2, lambda: self.client.set_drive(250, -700)),
            (1.5, self.client.stop_drive),
        ]

    def _demo_arm(self):
        return [
            (0.0, lambda: self.client.set_arm(self.HOME, 1000)),
            (1.2, lambda: self.client.set_arm([(0, 2200)], 700)),
            (0.9, lambda: self.client.set_arm([(1, 1250), (2, 2100), (3, 1000)], 900)),
            (1.1, lambda: self.client.set_arm([(5, 1500)], 400)),
            (0.6, lambda: self.client.set_arm(self.HOME, 1200)),
        ]

    def _demo_io(self):
        return [
            (0.0, lambda: self.client.set_led(LED_BLINK, 150)),
            (0.1, lambda: self.client.beep(3, 100, 100)),
            (1.2, lambda: self.client.set_led(LED_ON)),
            (0.8, lambda: self.client.set_led(LED_OFF)),
        ]

    def _demo_all(self):
        return self._demo_io() + self._demo_drive() + self._demo_arm()


def print_status(status):
    print(
        "uptime={uptime_ms}ms flags=0x{flags:04X} v={linear_mm_s}mm/s "
        "w={angular_mrad_s}mrad/s output=({left_output},{right_output}) "
        "wheels={wheel_feedback} joints={joint_positions} battery={battery_mv}mV "
        "bus_errors={bus_errors}".format(**status)
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: python tools/pi_command_console.py <串口设备>")
    client = RobotClient(sys.argv[1])
    console = CommandConsole(client)
    print("已连接。输入 help；输入 cancel 可随时中止动作；输入 quit 退出。")
    try:
        while True:
            try:
                if not console.execute(input("car> ")):
                    break
            except (ValueError, RuntimeError, TimeoutError) as exc:
                print("错误:", exc)
    finally:
        console.sequence.cancel(send_to_robot=True)
        client.close()


if __name__ == "__main__":
    main()
