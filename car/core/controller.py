"""树莓派命令入口、ESP32 安全状态机和状态汇总。"""

from core.timebase import ticks_diff, ticks_ms
from protocol.frame import (
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ERROR,
    FLAG_RESPONSE,
    FrameParser,
    encode_frame,
)
from protocol.ascii_commands import (
    CMD_ARM_DELTA,
    CMD_BEEP,
    CMD_LED,
    CMD_MOVE_DELTA,
    CMD_PING,
    CMD_STOP,
    parse_ascii_command,
)
from protocol.messages import (
    ERR_BAD_DESTINATION,
    ERR_BAD_VALUE,
    ERR_INTERNAL,
    ERR_UNSUPPORTED,
    MSG_ACK,
    MSG_ARM_STOP,
    MSG_BEEP,
    MSG_CANCEL,
    MSG_DRIVE_CALIBRATION,
    MSG_ERROR,
    MSG_HEARTBEAT,
    MSG_QUERY_DRIVE_CALIBRATION,
    MSG_QUERY_STATUS,
    MSG_RESET_DRIVE_CALIBRATION,
    MSG_ROBOT_STATUS,
    MSG_SET_ARM_JOINTS,
    MSG_SET_DRIVE_CALIBRATION,
    MSG_SET_LED,
    MSG_SET_BALANCED_TWIST,
    MSG_SET_TWIST,
    MSG_STOP,
    CANCEL_ARM,
    CANCEL_BUZZER,
    CANCEL_DRIVE,
    CANCEL_LED,
    decode_beep,
    decode_led,
    decode_arm_joints,
    decode_balanced_twist,
    decode_drive_calibration,
    decode_twist,
    encode_ack,
    encode_drive_calibration,
    encode_error,
    encode_robot_status,
)


# ==================== 状态标志位定义 ====================
# 机器人的运行状态使用位掩码（bitmask）表示，每个位代表一种状态
STATUS_DRIVE_MOVING = 0x0001        # 底盘正在移动
STATUS_ARM_MOVING = 0x0002          # 机械臂正在运动
STATUS_CLOSED_LOOP = 0x0004         # 底盘处于闭环控制模式（带反馈）
STATUS_PI_LINK = 0x0008             # 与树莓派的通信链路正常
STATUS_ACTUATOR_FAULT = 0x0010      # 执行器（电机/舵机）发生故障
STATUS_FAILSAFE = 0x0020            # 触发了安全保护模式（通信超时）
STATUS_LED_ON = 0x0040              # LED 指示灯当前点亮
STATUS_BUZZER_ACTIVE = 0x0080       # 蜂鸣器正在鸣响


class RobotController:
    """
    机器人主控制器类
    
    职责：
    1. 统一管理底盘(drive)、机械臂(arm)、指示灯(indicators)和总线(bus)
    2. 处理来自树莓派的各种命令（二进制协议和ASCII协议）
    3. 维护机器人状态（故障检测、通信超时保护、状态汇总）
    4. 实现安全机制：通信中断时自动停止并进入故障保护模式
    """
    
    def __init__(self, drive, arm, indicators, actuator_bus, cfg):
        """
        初始化机器人控制器
        
        参数:
            drive: 底盘驱动控制器对象
            arm: 机械臂控制器对象
            indicators: LED和蜂鸣器控制器对象
            actuator_bus: 执行器通信总线对象（与电机/舵机通信）
            cfg: 配置对象，包含各种超时、速度限制等参数
        """
        self.drive = drive
        self.arm = arm
        self.indicators = indicators
        self.bus = actuator_bus
        self.cfg = cfg
        self.started_ms = ticks_ms()                    # 系统启动时间（毫秒）
        self.last_pi_frame_ms = self.started_ms         # 最后一次收到树莓派帧的时间
        self.peer_seen = False                          # 是否曾收到过树莓派的消息
        self.failsafe_latched = False                   # 故障保护是否已锁定
        self.battery_mv = 0                             # 电池电压（毫伏）
        self.actuator_fault_flags = 0                   # 执行器故障标志位
        self.bus.on_status = self._on_actuator_status   # 注册总线状态回调

    def note_pi_activity(self, now=None):
        """
        记录树莓派的活动时间戳
        
        当收到来自树莓派的任何有效帧时调用，用于监控通信链路是否存活。
        同时清除故障保护锁定状态。
        """
        self.last_pi_frame_ms = ticks_ms() if now is None else now
        self.peer_seen = True
        self.failsafe_latched = False

    def handle_command(self, msg_type, payload, now=None):
        """
        处理二进制协议命令（来自树莓派的标准化消息帧）
        
        这是核心命令分发器，根据消息类型调用对应的处理逻辑。
        支持的命令类型：
            - MSG_HEARTBEAT: 心跳包，保持通信链路活跃
            - MSG_SET_TWIST: 设置底盘运动速度（线速度+角速度）
            - MSG_STOP: 紧急停止
            - MSG_CANCEL: 取消特定操作（驱动/机械臂/蜂鸣器/LED）
            - MSG_SET_ARM_JOINTS: 设置机械臂关节角度
            - MSG_ARM_STOP: 停止机械臂
            - MSG_SET_LED: 设置LED模式
            - MSG_BEEP: 蜂鸣器鸣响
            - MSG_QUERY_STATUS: 查询机器人状态
        
        返回:
            (响应消息类型, 编码后的响应载荷)
        
        异常:
            LookupError: 不支持的消息类型
            ValueError: 载荷格式错误
        """
        now = ticks_ms() if now is None else now
        
        # 心跳：保持连接，返回确认
        if msg_type == MSG_HEARTBEAT:
            if len(payload) not in (0, 4):
                raise ValueError("heartbeat payload must be empty or u32")
            return MSG_ACK, encode_ack(msg_type)
        
        # 设置底盘速度（带时间限制的运动指令）
        if msg_type == MSG_SET_TWIST:
            linear, angular, ttl_ms = decode_twist(payload)
            self.drive.set_twist(linear, angular, ttl_ms, now)
            return MSG_ACK, encode_ack(msg_type)

        if msg_type == MSG_SET_BALANCED_TWIST:
            linear, angular, ttl_ms, gains = decode_balanced_twist(
                payload
            )
            self.drive.set_twist(
                linear,
                angular,
                ttl_ms,
                now,
                motor_output_gains=gains,
            )
            return MSG_ACK, encode_ack(msg_type)
        
        # 紧急停止底盘
        if msg_type == MSG_STOP:
            if payload:
                raise ValueError("stop payload must be empty")
            self.drive.stop(emergency=True)
            return MSG_ACK, encode_ack(msg_type)
        
        # 取消指定的操作（位掩码控制）
        if msg_type == MSG_CANCEL:
            if len(payload) != 1:
                raise ValueError("cancel payload must contain one mask byte")
            self.cancel(payload[0])
            return MSG_ACK, encode_ack(msg_type)
        
        # 设置机械臂关节目标位置
        if msg_type == MSG_SET_ARM_JOINTS:
            self.arm.set_joints(decode_arm_joints(payload))
            return MSG_ACK, encode_ack(msg_type)
        
        # 停止机械臂
        if msg_type == MSG_ARM_STOP:
            if payload:
                raise ValueError("arm stop payload must be empty")
            self.arm.stop()
            return MSG_ACK, encode_ack(msg_type)
        
        # 设置LED指示灯模式
        if msg_type == MSG_SET_LED:
            mode, period_ms = decode_led(payload)
            self.indicators.set_led(mode, period_ms, now)
            return MSG_ACK, encode_ack(msg_type)
        
        # 蜂鸣器鸣响控制
        if msg_type == MSG_BEEP:
            repeat, on_ms, off_ms = decode_beep(payload)
            self.indicators.beep(repeat, on_ms, off_ms, now)
            return MSG_ACK, encode_ack(msg_type)
        
        # 查询机器人完整状态
        if msg_type == MSG_QUERY_STATUS:
            if payload:
                raise ValueError("status query payload must be empty")
            return MSG_ROBOT_STATUS, encode_robot_status(self.status(now))

        # 车辆必须停车后才允许更新，并原子保存到 drive_calibration.json。
        if msg_type == MSG_SET_DRIVE_CALIBRATION:
            values = decode_drive_calibration(payload)
            self.drive.set_calibration(
                values["trim_intercept"],
                values["trim_slope_per_mm_s"],
            )
            return MSG_ACK, encode_ack(msg_type)

        if msg_type == MSG_QUERY_DRIVE_CALIBRATION:
            if payload:
                raise ValueError("calibration query payload must be empty")
            intercept, slope = self.drive.calibration_values()
            return MSG_DRIVE_CALIBRATION, encode_drive_calibration(
                intercept, slope
            )

        if msg_type == MSG_RESET_DRIVE_CALIBRATION:
            if payload:
                raise ValueError("calibration reset payload must be empty")
            self.drive.reset_calibration()
            return MSG_ACK, encode_ack(msg_type)
        
        raise LookupError("unsupported message type")

    def handle_ascii_command(self, text, now=None):
        """
        处理ASCII文本协议命令（用于简单调试和人工控制）
        
        支持的命令（以#开头，!结尾）：
            - PING: 心跳测试，返回 OK:PING
            - STOP: 停止所有运动，取消所有操作
            - ARM_DELTA: 机械臂增量运动（前后/上下/旋转/夹爪）
            - MOVE_DELTA: 底盘增量运动（前进距离/转向角度/速度等级）
            - LED: 设置LED模式
            - BEEP: 蜂鸣器鸣响
        
        返回:
            编码后的ASCII响应字符串
        """
        now = ticks_ms() if now is None else now
        command, args = parse_ascii_command(text)
        
        if command == CMD_PING:
            return b"OK:PING\n"
        
        if command == CMD_STOP:
            self.cancel(CANCEL_DRIVE | CANCEL_ARM | CANCEL_BUZZER | CANCEL_LED)
            return b"OK:STOP\n"
        
        if command == CMD_ARM_DELTA:
            duration = self.arm.move_cartesian_delta(
                args["forward_mm"], args["up_mm"], args["left_deg"],
                args["claw_open"],
            )
            return "OK:ARM,T{:04d}\n".format(duration).encode("ascii")
        
        if command == CMD_MOVE_DELTA:
            duration = self._drive_delta(
                args["forward_mm"], args["left_deg"], args["speed_level"], now
            )
            return "OK:MOV,T{:04d}\n".format(duration).encode("ascii")
        
        if command == CMD_LED:
            self.indicators.set_led(args["mode"], args["period_ms"], now)
            return b"OK:LED\n"
        
        if command == CMD_BEEP:
            repeat = min(max(args["repeat"], 1), 9)
            self.indicators.beep(repeat, 120, 120, now)
            return b"OK:BEEP\n"
        
        raise LookupError("unsupported ASCII command")

    def cancel(self, mask):
        """
        按位掩码取消指定操作
        
        参数:
            mask: 位掩码，可包含 CANCEL_DRIVE | CANCEL_ARM | CANCEL_BUZZER | CANCEL_LED
        """
        if mask & CANCEL_DRIVE:
            self.drive.stop(emergency=True)
        if mask & CANCEL_ARM:
            self.arm.stop()
        if mask & CANCEL_BUZZER:
            self.indicators.cancel_buzzer()
        if mask & CANCEL_LED:
            self.indicators.cancel_led()

    def update(self, now=None):
        """
        周期性更新函数（在主循环中调用）
        
        职责：
        1. 检查与树莓派的通信超时（超时则触发故障保护）
        2. 更新底盘（执行运动插值）
        3. 更新机械臂（执行关节插值）
        4. 更新指示灯/蜂鸣器（闪烁/鸣响计时）
        """
        now = ticks_ms() if now is None else now
        
        # 通信超时检测：如果超过 PI_LINK_TIMEOUT_MS 未收到树莓派消息
        if (ticks_diff(now, self.last_pi_frame_ms) > self.cfg.PI_LINK_TIMEOUT_MS and
                not self.failsafe_latched):
            # 安全保护：紧急停止底盘和机械臂，关闭蜂鸣器
            self.drive.stop(emergency=True)
            self.arm.stop()
            self.indicators.cancel_buzzer()
            self.failsafe_latched = True  # 锁定故障状态，防止反复触发
        
        # 更新各子模块
        self.drive.update(now)
        if hasattr(self.arm, "update"):
            self.arm.update(now)
        self.indicators.update(now)

    def link_alive(self, now=None):
        """
        检查与树莓派的通信链路是否存活
        
        返回:
            True: 链路正常（在超时时间内收到过数据）
            False: 链路超时
        """
        now = ticks_ms() if now is None else now
        return self.peer_seen and ticks_diff(now, self.last_pi_frame_ms) <= self.cfg.PI_LINK_TIMEOUT_MS

    def status(self, now=None):
        """
        收集并汇总机器人当前完整状态
        
        返回:
            包含以下字段的字典：
                uptime_ms: 系统运行时间（毫秒）
                flags: 状态标志位组合（见文件开头的 STATUS_* 定义）
                linear_mm_s: 当前线速度（mm/s）
                angular_mrad_s: 当前角速度（mrad/s）
                left_output: 左轮输出值
                right_output: 右轮输出值
                wheel_feedback: 轮子编码器反馈
                joint_positions: 机械臂各关节位置
                battery_mv: 电池电压（毫伏）
                bus_errors: 总线错误计数
        """
        now = ticks_ms() if now is None else now
        flags = 0
        if self.drive.moving:
            flags |= STATUS_DRIVE_MOVING
        if self.arm.moving:
            flags |= STATUS_ARM_MOVING
        if self.drive.closed_loop:
            flags |= STATUS_CLOSED_LOOP
        if self.link_alive(now):
            flags |= STATUS_PI_LINK
        if self.actuator_fault_flags or self.bus.error_count:
            flags |= STATUS_ACTUATOR_FAULT
        if self.failsafe_latched:
            flags |= STATUS_FAILSAFE
        if self.indicators.led_value:
            flags |= STATUS_LED_ON
        if self.indicators.buzzer_active:
            flags |= STATUS_BUZZER_ACTIVE
        
        return {
            "uptime_ms": max(0, ticks_diff(now, self.started_ms)),
            "flags": flags,
            "linear_mm_s": int(self.drive.current_linear),
            "angular_mrad_s": int(self.drive.current_angular),
            "left_output": int(self.drive.outputs[0]),
            "right_output": int(self.drive.outputs[1]),
            "wheel_feedback": self.drive.feedback,
            "joint_positions": self.arm.positions,
            "battery_mv": self.battery_mv,
            "bus_errors": min(self.bus.error_count, 0xFFFF),
        }

    def _on_actuator_status(self, _source, status, now):
        """
        执行器状态回调函数（由 actuator_bus 调用）
        
        当执行器总线返回状态更新时，此方法被调用，用于：
        1. 更新轮子的编码器反馈
        2. 更新机械臂关节反馈
        3. 更新电池电压
        4. 更新故障标志
        """
        self.drive.set_feedback(status["wheels"], now)
        self.arm.set_feedback(status["joints"])
        self.battery_mv = status["battery_mv"]
        self.actuator_fault_flags = status["fault_flags"]

    def _drive_delta(self, forward_mm, left_deg, speed_level, now):
        """
        执行增量式底盘运动（ASCII命令专用）
        
        将前进距离和转向角度转换为带时限的速度指令。
        
        参数:
            forward_mm: 前进距离（毫米，正数向前，负数向后）
            left_deg: 左转角度（度，正数左转，负数右转）
            speed_level: 速度等级（1-100），百分比
            now: 当前时间戳
        
        返回:
            运动持续时间（毫秒）
        """
        distance = abs(int(forward_mm))
        angle = abs(int(left_deg))
        if distance == 0 and angle == 0:
            self.drive.stop(emergency=True)
            return 0

        # 速度等级限制在 1-100
        level = int(speed_level) if speed_level is not None else getattr(
            self.cfg, "ASCII_MOVE_DEFAULT_SPEED_LEVEL", 70
        )
        level = int(_clamp(level, 1, 100))
        
        # 根据速度等级计算实际速度
        linear_speed = self.cfg.MAX_LINEAR_MM_S * level / 100.0
        angular_speed = self.cfg.MAX_ANGULAR_MRAD_S * level / 100.0

        # 分别计算直线运动和旋转运动所需时间，取较大值
        linear_seconds = distance / max(linear_speed, 1.0) if distance else 0.0
        angle_mrad = angle * 3.1415926 / 180.0 * 1000.0  # 角度转毫弧度
        angular_seconds = angle_mrad / max(angular_speed, 1.0) if angle else 0.0
        
        # 持续时间限制在最小时间（保证至少执行一段时间）和最大时间之间
        duration = int(max(
            getattr(self.cfg, "ASCII_MOVE_MIN_DURATION_MS", 450),
            min(self.cfg.COMMAND_MAX_TTL_MS,
                max(linear_seconds, angular_seconds) * 1000.0),
        ))

        # 计算带符号的速度值
        linear_sign = 1 if forward_mm >= 0 else -1
        angular_sign = 1 if left_deg >= 0 else -1
        linear = int(_clamp(
            linear_sign * linear_speed if distance else 0,
            -self.cfg.MAX_LINEAR_MM_S,
            self.cfg.MAX_LINEAR_MM_S,
        ))
        angular = int(_clamp(
            angular_sign * angular_speed if angle else 0,
            -self.cfg.MAX_ANGULAR_MRAD_S,
            self.cfg.MAX_ANGULAR_MRAD_S,
        ))
        
        self.drive.set_twist(linear, angular, duration, now)
        return duration


class PiCommandService:
    """
    树莓派命令服务类
    
    职责：
    1. 管理专用 UART 串口通信（与树莓派之间的物理链路）
    2. 解析接收到的数据（区分二进制帧和ASCII文本命令）
    3. 处理重放攻击防护（幂等性缓存）
    4. 地址校验（只接受来自树莓派且目标为ESP32的消息）
    5. 错误处理和异常响应
    """
    
    def __init__(self, transport, controller):
        """
        初始化命令服务
        
        参数:
            transport: 串口传输对象（负责物理层读写）
            controller: 机器人控制器实例
        """
        self.transport = transport
        self.controller = controller
        self.parser = FrameParser()              # 二进制帧解析器
        self.bad_source_count = 0                # 来源地址错误的计数
        self.bad_destination_count = 0           # 目标地址错误的计数
        self._cache = []                         # 命令响应缓存（用于幂等性）
        self._tx_seq = 0                         # 发送序列号（自增）
        self._ascii_buffer = bytearray()         # ASCII命令缓冲区（跨包拼接）

    def poll(self, now=None):
        """
        轮询函数（在主循环中周期性调用）
        
        从串口读取数据，自动识别并处理二进制帧和ASCII命令。
        支持两种协议混合传输。
        """
        now = ticks_ms() if now is None else now
        data = self.transport.read()
        if not data:
            return
        
        # 先处理 ASCII 命令（# ... !），剩余数据交给二进制解析器
        binary = self._handle_ascii_bytes(data, now)
        for frame in self.parser.feed(binary):
            self._handle_frame(frame, now)

    def send_status(self, now=None):
        """
        主动向树莓派发送机器人状态（心跳响应/定时上报）
        
        将当前状态编码为二进制帧并通过串口发送。
        """
        payload = encode_robot_status(self.controller.status(now))
        raw = encode_frame(
            MSG_ROBOT_STATUS, self._next_tx_seq(), payload,
            src=ADDR_ESP32, dst=ADDR_RASPBERRY_PI,
        )
        self.transport.write(raw)

    def _handle_frame(self, frame, now):
        """
        处理单个二进制帧
        
        包含完整的请求验证、命令分发、响应缓存流程：
        1. 地址校验（只处理树莓派→ESP32的帧）
        2. 幂等性检查（重放命令直接返回缓存的响应）
        3. 更新通信时间戳
        4. 调用控制器执行命令
        5. 缓存响应并发送
        """
        # 地址校验：只接受来自树莓派的帧
        if frame.src != ADDR_RASPBERRY_PI:
            self.bad_source_count += 1
            return
        
        # 地址校验：目标必须是ESP32
        if frame.dst != ADDR_ESP32:
            self.bad_destination_count += 1
            self._send_error(frame, ERR_BAD_DESTINATION)
            return
        
        # 忽略响应帧（只处理请求帧）
        if frame.is_response:
            return

        # 幂等性检查：如果收到重复命令，直接返回缓存的响应
        key = (frame.seq, frame.msg_type, frame.payload)
        for cached_key, response in self._cache:
            if key == cached_key:
                self.transport.write(response)
                return

        # 记录树莓派活动时间
        self.controller.note_pi_activity(now)
        
        # 执行命令（捕获各种异常）
        try:
            response_type, payload = self.controller.handle_command(
                frame.msg_type, frame.payload, now
            )
            flags = FLAG_RESPONSE
        except LookupError:
            # 不支持的命令类型
            response_type = MSG_ERROR
            payload = encode_error(frame.msg_type, ERR_UNSUPPORTED)
            flags = FLAG_RESPONSE | FLAG_ERROR
        except (ValueError, TypeError):
            # 参数格式错误
            response_type = MSG_ERROR
            payload = encode_error(frame.msg_type, ERR_BAD_VALUE)
            flags = FLAG_RESPONSE | FLAG_ERROR
        except Exception as exc:
            # 内部错误（打印调试信息）
            print("controller error:", exc)
            response_type = MSG_ERROR
            payload = encode_error(frame.msg_type, ERR_INTERNAL)
            flags = FLAG_RESPONSE | FLAG_ERROR

        # 构造响应帧并发送
        raw = encode_frame(
            response_type, frame.seq, payload,
            src=ADDR_ESP32, dst=ADDR_RASPBERRY_PI, flags=flags,
        )
        self.transport.write(raw)
        
        # 缓存响应（最多8条，FIFO淘汰）
        self._cache.append((key, raw))
        if len(self._cache) > 8:
            self._cache.pop(0)

    def _send_error(self, frame, error_code):
        """
        发送错误响应帧
        
        用于地址校验失败等场景，向树莓派报告错误原因。
        """
        raw = encode_frame(
            MSG_ERROR, frame.seq, encode_error(frame.msg_type, error_code),
            src=ADDR_ESP32, dst=ADDR_RASPBERRY_PI,
            flags=FLAG_RESPONSE | FLAG_ERROR,
        )
        self.transport.write(raw)

    def _next_tx_seq(self):
        """获取下一个发送序列号（16位循环自增）"""
        value = self._tx_seq
        self._tx_seq = (self._tx_seq + 1) & 0xFFFF
        return value

    def _handle_ascii_bytes(self, data, now):
        """
        从数据流中提取 ASCII 命令
        
        ASCII 命令格式：#命令内容!
        遇到 # 开始缓存，遇到 ! 表示命令结束并立即处理。
        非 ASCII 数据（二进制帧）原样返回给调用者。
        
        返回:
            过滤掉 ASCII 命令后的二进制数据
        """
        binary = bytearray()
        for byte in data:
            if self._ascii_buffer:
                # 正在缓存 ASCII 命令
                self._ascii_buffer.append(byte)
                if byte == ord("!"):
                    # 命令结束，执行处理
                    self._handle_ascii_frame(bytes(self._ascii_buffer), now)
                    self._ascii_buffer = bytearray()
                elif len(self._ascii_buffer) > 64:
                    # 防止缓冲区溢出（超过64字节丢弃）
                    self._ascii_buffer = bytearray()
            elif byte == ord("#"):
                # 开始新的 ASCII 命令
                self._ascii_buffer.append(byte)
            else:
                # 非 ASCII 数据，留给二进制解析器
                binary.append(byte)
        return bytes(binary)

    def _handle_ascii_frame(self, raw, now):
        """
        处理完整的 ASCII 命令帧
        
        调用控制器的 handle_ascii_command 执行命令并返回响应。
        """
        self.controller.note_pi_activity(now)
        try:
            response = self.controller.handle_ascii_command(raw, now)
        except Exception as exc:
            response = ("ERR:{}\n".format(exc)).encode("ascii")
        self.transport.write(response)


def _clamp(value, minimum, maximum):
    """数值限制函数，将 value 限制在 [minimum, maximum] 范围内"""
    return min(max(value, minimum), maximum)
