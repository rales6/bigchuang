"""Thonny / MicroPython 单线程手动与自动测试工具。

不要在 Thonny REPL 中用后台线程打印：异步输出会干扰 Thonny 的 raw-REPL
控制字符。本模块让命令、控制循环和输出都在当前 Shell 调用中同步完成。

快速使用：
    import thonny_manual_test as test
    test.send("beep_3")
    test.run_auto_test("io")

车轮悬空、机械臂周围无障碍后：
    test.run_auto_test("all")
"""

from app import ExecutorApplication
from core.timebase import sleep_ms, ticks_add, ticks_diff, ticks_ms
from protocol.messages import (
    CANCEL_ALL,
    CANCEL_ARM,
    CANCEL_BUZZER,
    CANCEL_DRIVE,
    LED_BLINK,
    LED_OFF,
    LED_ON,
    MSG_BEEP,
    MSG_CANCEL,
    MSG_HEARTBEAT,
    MSG_QUERY_STATUS,
    MSG_SET_ARM_JOINTS,
    MSG_SET_LED,
    MSG_SET_TWIST,
    decode_robot_status,
    encode_arm_joints,
    encode_beep,
    encode_led,
    encode_twist,
)

TEST_MODULE_VERSION = "2026.07.14-uart-all-v10"


class _NullTransport:
    name = "thonny-manual"

    def read(self):
        return b""

    def write(self, data):
        return len(data)

    def deinit(self):
        pass


_app = None


def start():
    """初始化硬件；send/run_auto_test 也会自动调用。"""
    global _app
    if _app is None:
        _app = ExecutorApplication(pi_transport=_NullTransport())
        print("manual test ready:", TEST_MODULE_VERSION)
    return _app


def send(command, observe_ms=None):
    """同步执行一条命令，完成后才返回 Thonny 提示符。

    observe_ms 可覆盖默认观察时间。运动类命令在观察结束后自动停止，避免测试
    脚本结束或 Thonny 断开时继续运动。
    """
    app = start()
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    command = command.strip()
    try:
        default_ms, cleanup_mask = _execute(command)
        duration = default_ms if observe_ms is None else int(observe_ms)
        if duration < 0 or duration > 10000:
            raise ValueError("observe_ms must be in range 0..10000")
        _pump(duration, keep_link=True)
        if cleanup_mask:
            app.controller.cancel(cleanup_mask)
            _pump(160)
        print("completed:", command)
        return status(print_result=False)
    except KeyboardInterrupt:
        _safe_cancel()
        print("interrupted and cancelled")
        raise
    except Exception:
        _safe_cancel()
        raise


def send_ascii(command, observe_ms=None):
    """Run a Raspberry-Pi-style ASCII frame, for example #ARMF020U030L000!."""
    app = start()
    command = command.strip()
    now = ticks_ms()
    app.controller.note_pi_activity(now)
    response = app.controller.handle_ascii_command(command, now)
    print(response.decode("ascii").strip())
    duration = 1200 if observe_ms is None else int(observe_ms)
    _pump(duration, keep_link=True)
    return status(print_result=False)


def run_auto_test(scope="io"):
    """运行规范化自动测试。

    scope:
        io     LED、蜂鸣器和通信（默认，最安全）
        drive  四轮底盘；必须先把车轮悬空
        arm    机械臂；必须确保活动范围无障碍
        all    依次运行以上全部项目
    """
    scopes = ("io", "drive", "arm", "all")
    if scope not in scopes:
        raise ValueError("scope must be one of: io, drive, arm, all")

    start()
    groups = []
    if scope in ("io", "all"):
        groups.append(("IO", _IO_TESTS))
    if scope in ("drive", "all"):
        groups.append(("DRIVE", _DRIVE_TESTS))
    if scope in ("arm", "all"):
        groups.append(("ARM", _ARM_TESTS))

    passed = 0
    failed = 0
    print("=== AUTO TEST START: {} ===".format(scope))
    try:
        send("ping")
        for group_name, cases in groups:
            print("--- {} ---".format(group_name))
            for case_name, command, observe_ms in cases:
                print("[RUN] {} -> {}".format(case_name, command))
                try:
                    motor_writes_before = getattr(
                        _app.bus, "nonzero_drive_write_count", None
                    )
                    arm_writes_before = getattr(_app.arm, "write_count", None)
                    if arm_writes_before is None:
                        arm_writes_before = getattr(_app.bus, "arm_write_count", None)
                    send(command, observe_ms)
                    if (group_name == "DRIVE" and motor_writes_before is not None and
                            _app.bus.nonzero_drive_write_count <= motor_writes_before):
                        raise RuntimeError("no non-zero motor UART command was written")
                    if (group_name == "ARM" and command != "arm_home" and
                            arm_writes_before is not None):
                        current_arm_writes = getattr(_app.arm, "write_count", None)
                        if current_arm_writes is None:
                            current_arm_writes = getattr(_app.bus, "arm_write_count", None)
                        if current_arm_writes <= arm_writes_before:
                            raise RuntimeError("no servo UART command was written")
                    passed += 1
                    print("[PASS] driver output confirmed:", case_name)
                    _pump(500, keep_link=True)
                except Exception as exc:
                    failed += 1
                    print("[FAIL] {}: {}".format(case_name, exc))
                    _safe_cancel()
                    # 执行器异常后停止本组，避免继续动作扩大风险。
                    break
        final_status = status(print_result=True)
        diagnostics()
        if final_status["bus_errors"]:
            failed += 1
            print("[FAIL] actuator bus errors:", final_status["bus_errors"])
    finally:
        _safe_cancel()
    print("=== AUTO TEST END: passed={}, failed={} ===".format(passed, failed))
    return failed == 0


def status(print_result=True):
    """读取当前汇总状态。"""
    app = start()
    now = ticks_ms()
    app.controller.note_pi_activity(now)
    _msg_type, payload = app.controller.handle_command(MSG_QUERY_STATUS, b"", now)
    result = decode_robot_status(payload)
    if print_result:
        print(result)
    return result


def diagnostics():
    """显示驱动层计数；用于区分“没有生成输出”和“外部硬件没有动作”。"""
    app = start()
    result = {
        "test_module": TEST_MODULE_VERSION,
        "backend": getattr(__import__("config"), "ACTUATOR_BACKEND", "unknown"),
        "firmware_build": getattr(__import__("config"), "FIRMWARE_BUILD", "unknown"),
        "motor_uart_writes": getattr(app.bus, "write_count", None),
        "motor_drive_writes": getattr(app.bus, "drive_write_count", None),
        "motor_nonzero_writes": getattr(
            app.bus, "nonzero_drive_write_count", None
        ),
        "motor_stop_writes": getattr(app.bus, "stop_write_count", None),
        "motor_rx_bytes": getattr(app.bus, "received_bytes", None),
        "servo_pwm_writes": getattr(app.arm, "write_count", None),
        "servo_uart_writes": getattr(app.bus, "arm_write_count", None),
        "servo_stop_writes": getattr(app.bus, "arm_stop_write_count", None),
        "last_motor_frame": getattr(app.bus, "last_write", b""),
        "last_drive_frame": getattr(app.bus, "last_drive_frame", b""),
        "last_nonzero_drive_frame": getattr(
            app.bus, "last_nonzero_drive_frame", b""
        ),
        "last_stop_frame": getattr(app.bus, "last_stop_frame", b""),
        "last_arm_frame": getattr(app.bus, "last_arm_frame", b""),
        "last_arm_stop_frame": getattr(app.bus, "last_arm_stop_frame", b""),
    }
    print(result)
    return result


def raw_motor_test(speed=150, duration_ms=400):
    """绕过运动控制层，直接发送原底板四轮字符串；车轮必须悬空。"""
    app = start()
    speed = int(speed)
    duration_ms = int(duration_ms)
    if not 50 <= abs(speed) <= 500:
        raise ValueError("raw motor speed absolute value must be in range 50..500")
    if not 100 <= duration_ms <= 1000:
        raise ValueError("duration_ms must be in range 100..1000")
    frame = (
        "#006P{:04d}T{:04d}!#007P{:04d}T{:04d}!"
        "#008P{:04d}T{:04d}!#009P{:04d}T{:04d}!"
    ).format(
        1500 + speed, duration_ms, 1500 - speed, duration_ms,
        1500 + speed, duration_ms, 1500 - speed, duration_ms,
    ).encode("ascii")
    stop_frame = (
        b"#006P1500T0100!#007P1500T0100!"
        b"#008P1500T0100!#009P1500T0100!"
    )
    print("raw TX:", frame)
    app.actuator_transport.write(frame)
    try:
        sleep_ms(duration_ms + 80)
    finally:
        app.actuator_transport.write(stop_frame)
    print("raw motor test completed and stopped")


def raw_servo_test(joint_id=0, pulse_us=1700, duration_ms=500):
    """绕过机械臂控制层，直接发送一个 #000–#005 舵机字符串。"""
    app = start()
    joint_id = int(joint_id)
    pulse_us = int(pulse_us)
    duration_ms = int(duration_ms)
    if not 0 <= joint_id < 6:
        raise ValueError("joint_id must be in range 0..5")
    minimum, maximum = __import__("config").ARM_LIMITS_US[joint_id]
    if not minimum <= pulse_us <= maximum:
        raise ValueError("pulse exceeds configured joint limit")
    if not 100 <= duration_ms <= 2000:
        raise ValueError("duration_ms must be in range 100..2000")
    frame = "#{:03d}P{:04d}T{:04d}!".format(
        joint_id, pulse_us, duration_ms
    ).encode("ascii")
    print("raw TX:", frame)
    app.actuator_transport.write(frame)
    sleep_ms(duration_ms + 80)
    print("raw servo test completed")


def cancel():
    """立即停止所有测试动作。"""
    start()
    _safe_cancel()
    print("all actions cancelled")


def stop():
    """停止全部动作、释放 UART，并允许之后重新初始化。"""
    global _app
    if _app is None:
        return
    _safe_cancel()
    try:
        _app.bus.stop(mask=0x03)
        _pump(80)
        _app.indicators.cancel_all()
        if hasattr(_app.arm, "deinit"):
            _app.arm.deinit()
        _app.actuator_transport.deinit()
    finally:
        _app = None
    print("manual test closed")


def help():
    print(_HELP)


def _execute(line):
    parts = line.lower().split()
    name = parts[0]
    args = parts[1:]
    now = ticks_ms()
    _app.controller.note_pi_activity(now)

    if line.startswith("#") and line.endswith("!"):
        response = _app.controller.handle_ascii_command(line, now)
        print(response.decode("ascii").strip())
        return 1200, 0

    if name == "help":
        help()
        return 0, 0
    if name == "ping":
        _app.controller.handle_command(MSG_HEARTBEAT, b"", now)
        print("pong")
        return 20, 0
    if name == "status":
        status(print_result=True)
        return 0, 0
    if name == "diagnostics":
        diagnostics()
        return 0, 0
    if name in ("cancel", "stop_all"):
        _message(MSG_CANCEL, bytes((CANCEL_ALL,)), now)
        return 80, 0
    if name == "stop_drive":
        _message(MSG_CANCEL, bytes((CANCEL_DRIVE,)), now)
        return 80, 0
    if name == "arm_stop":
        _message(MSG_CANCEL, bytes((CANCEL_ARM,)), now)
        return 80, 0

    if name == "forward_slow":
        return _twist(180, 0, now)
    if name == "forward":
        return _twist(350, 0, now)
    if name == "backward":
        return _twist(-300, 0, now)
    if name == "spin_left":
        return _twist(0, 2200, now)
    if name == "spin_right":
        return _twist(0, -2200, now)
    if name == "curve_left":
        return _twist(300, 900, now)
    if name == "curve_right":
        return _twist(300, -900, now)
    if name == "drive":
        if len(args) not in (2, 3):
            raise ValueError("drive <linear_mm_s> <angular_mrad_s> [ttl_ms]")
        ttl_ms = int(args[2]) if len(args) == 3 else 2000
        return _twist(int(args[0]), int(args[1]), now, ttl_ms)

    if name == "arm_home":
        return _arm([
            (0, 1500), (1, 1700), (2, 2000),
            (3, 1100), (4, 1500), (5, 1200),
        ], 2200, now)
    if name == "arm_left":
        return _arm([(0, 2200)], 2200, now)
    if name == "arm_right":
        return _arm([(0, 800)], 2800, now)
    if name == "arm_up":
        return _arm([(1, 1600), (2, 1700), (3, 1350)], 2200, now)
    if name == "arm_down":
        return _arm([(1, 1200), (2, 2100), (3, 1000)], 2200, now)
    if name == "grip":
        return _arm([(5, 1500)], 1400, now)
    if name == "release":
        return _arm([(5, 1200)], 1400, now)
    if name == "arm":
        if len(args) not in (2, 3):
            raise ValueError("arm <joint_id> <pulse_us> [duration_ms]")
        duration = int(args[2]) if len(args) == 3 else 1500
        return _arm([(int(args[0]), int(args[1]))], duration, now)

    if name == "led_off":
        _message(MSG_SET_LED, encode_led(LED_OFF), now)
        return 100, 0
    if name == "led_on":
        _message(MSG_SET_LED, encode_led(LED_ON), now)
        return 300, 0
    if name == "led_blink_slow":
        _message(MSG_SET_LED, encode_led(LED_BLINK, 500), now)
        return 1200, 0
    if name == "led_blink_fast":
        _message(MSG_SET_LED, encode_led(LED_BLINK, 100), now)
        return 600, 0
    if name == "beep_short":
        _message(MSG_BEEP, encode_beep(1, 100, 100), now)
        return 250, CANCEL_BUZZER
    if name == "beep_3":
        _message(MSG_BEEP, encode_beep(3, 100, 100), now)
        return 700, CANCEL_BUZZER

    raise ValueError("unknown command: {}".format(name))


def _twist(linear, angular, now, ttl_ms=2000):
    _app.controller.cancel(CANCEL_DRIVE)
    _message(MSG_SET_TWIST, encode_twist(linear, angular, ttl_ms), now)
    # 在 TTL 前结束观察并主动停止，不依赖超时兜底。
    return min(max(ttl_ms - 200, 100), 1800), CANCEL_DRIVE


def _arm(joints, duration_ms, now):
    _message(MSG_SET_ARM_JOINTS, encode_arm_joints(joints, duration_ms), now)
    # 舵机位置帧有明确的运动时间，正常完成后无需再发 #255PDST!。
    # 显式 arm_stop、Ctrl+C、异常以及真正的中途新命令仍会立即抢占。
    return min(duration_ms + 500, 10000), 0


def _message(msg_type, payload, now):
    return _app.controller.handle_command(msg_type, payload, now)


def _pump(duration_ms, keep_link=False):
    """在当前 Thonny 调用中持续推进通信和控制，不产生异步输出。"""
    deadline = ticks_add(ticks_ms(), duration_ms)
    while ticks_diff(deadline, ticks_ms()) > 0:
        if keep_link:
            _app.controller.note_pi_activity()
        _app.step()
        sleep_ms(5)


def _safe_cancel():
    if _app is None:
        return
    _app.controller.cancel(CANCEL_ALL)
    # 给最高优先级停止帧留出发送机会。
    try:
        _pump(180)
    except Exception:
        pass


_IO_TESTS = (
    ("LED 常亮", "led_on", 300),
    ("LED 慢闪", "led_blink_slow", 1200),
    ("LED 快闪", "led_blink_fast", 600),
    ("LED 关闭", "led_off", 100),
    ("蜂鸣器短响", "beep_short", 250),
    ("蜂鸣器三响", "beep_3", 700),
)

_DRIVE_TESTS = (
    ("低速前进", "forward_slow", 1500),
    ("后退", "backward", 1500),
    ("原地左转", "spin_left", 1500),
    ("原地右转", "spin_right", 1500),
    ("左弧线", "curve_left", 1500),
    ("右弧线", "curve_right", 1500),
)

_ARM_TESTS = (
    ("机械臂回零", "arm_home", 2600),
    ("底座左转", "arm_left", 2600),
    ("底座右转", "arm_right", 3200),
    ("机械臂抬升", "arm_up", 2600),
    ("机械臂下降", "arm_down", 2600),
    ("夹爪闭合", "grip", 1800),
    ("夹爪松开", "release", 1800),
    ("机械臂回零", "arm_home", 2600),
)

_HELP = """commands:
  ping, status, diagnostics, cancel, stop_all
  forward_slow, forward, backward, spin_left, spin_right
  curve_left, curve_right, drive <v> <w> [ttl], stop_drive
  arm_home, arm_left, arm_right, arm_up, arm_down
  grip, release, arm <id> <pulse> [duration], arm_stop
  led_on, led_off, led_blink_slow, led_blink_fast
  beep_short, beep_3

automatic tests:
  run_auto_test("io")
  run_auto_test("drive")  # wheels must be lifted
  run_auto_test("arm")    # clear arm workspace
  run_auto_test("all")    # wheels lifted + clear arm workspace

direct UART hardware tests:
  raw_motor_test(150, 400)       # wheels must be lifted
  raw_servo_test(0, 1700, 500)   # clear arm workspace
  send("#ARMF020U030L000!")      # Raspberry-Pi-style ASCII command
"""
