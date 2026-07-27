"""ESP32 执行器固件装配与协作式主循环。"""

import config
from core.controller import PiCommandService, RobotController
from core.timebase import sleep_ms, ticks_diff, ticks_ms
from drivers.actuator_bus import ActuatorBus
from drivers.arm import ArmController
from drivers.drive_base import DriveBase
from drivers.drive_calibration import DriveCalibration
from drivers.indicators import IndicatorController
from drivers.legacy_motor_bus import LegacyMotorBus
from drivers.pwm_arm import create_pwm_arm
from transport.uart import UARTTransport
from transport.multiplex import MultiplexTransport


class ExecutorApplication:
    def __init__(self, pi_transport=None, actuator_transport=None, indicators=None,
                 actuator_bus=None, arm=None):
        self.pi_transport = pi_transport or self._create_pi_transport()
        self.actuator_transport = actuator_transport or self._create_uart(
            config.ACTUATOR_UART_ID, config.ACTUATOR_UART_BAUDRATE,
            config.ACTUATOR_UART_TX_PIN, config.ACTUATOR_UART_RX_PIN,
            config.ACTUATOR_UART_RX_BUFFER, "actuator-bus",
        )
        self.bus = actuator_bus or self._create_actuator_bus()
        self.drive_calibration = DriveCalibration.load(
            config.DRIVE_CALIBRATION_PATH
        )
        self.drive = DriveBase(self.bus, config, self.drive_calibration)
        self.arm = arm or self._create_arm()
        self.indicators = indicators or self._create_indicators()
        self.controller = RobotController(
            self.drive, self.arm, self.indicators, self.bus, config
        )
        self.pi_service = PiCommandService(self.pi_transport, self.controller)
        self._last_status_ms = ticks_ms()

    def _create_actuator_bus(self):
        if config.ACTUATOR_BACKEND in ("legacy_uart_all", "legacy_uart_pwm"):
            return LegacyMotorBus(
                self.actuator_transport,
                config.LEGACY_MOTOR_IDS,
                config.LEGACY_MOTOR_PROTOCOL_SIGNS,
                config.LEGACY_MOTOR_COMMAND_TIME_MS,
            )
        if config.ACTUATOR_BACKEND == "vehicle_link_v2":
            return ActuatorBus(
                self.actuator_transport, config.ACTUATOR_NODE_ADDRESSES
            )
        raise ValueError("unsupported ACTUATOR_BACKEND: {}".format(
            config.ACTUATOR_BACKEND
        ))

    def _create_pi_transport(self):
        uart = self._create_uart(
            config.PI_UART_ID, config.PI_UART_BAUDRATE,
            config.PI_UART_TX_PIN, config.PI_UART_RX_PIN,
            config.PI_UART_RX_BUFFER, "uart",
        )
        if not getattr(config, "PI_BLE_ENABLED", False):
            return uart
        try:
            from transport.bluetooth import BLEUARTTransport
            ble = BLEUARTTransport(config.PI_BLE_NAME, config.PI_BLE_RX_BUFFER)
            return MultiplexTransport((uart, ble))
        except Exception as exc:
            # BLE 不可用不能阻止主 UART 安全链路启动。
            print("BLE backup unavailable:", exc)
            return uart

    def _create_arm(self):
        if config.ACTUATOR_BACKEND == "legacy_uart_all":
            return ArmController(self.bus, config)
        if config.ACTUATOR_BACKEND == "legacy_uart_pwm":
            return create_pwm_arm(config)
        return ArmController(self.bus, config)

    @staticmethod
    def _create_uart(uart_id, baudrate, tx_pin, rx_pin, rx_buffer, name):
        from machine import Pin, UART
        kwargs = {
            "baudrate": baudrate,
            "bits": 8,
            "parity": None,
            "stop": 1,
            "tx": Pin(tx_pin),
            "rx": Pin(rx_pin),
            "rxbuf": rx_buffer,
        }
        try:
            uart = UART(uart_id, **kwargs)
        except TypeError:
            kwargs.pop("rxbuf")
            uart = UART(uart_id, **kwargs)
        return UARTTransport(uart, name)

    @staticmethod
    def _create_indicators():
        from machine import Pin
        led = Pin(config.LED_PIN, Pin.OUT)
        buzzer = Pin(config.BUZZER_PIN, Pin.OUT)

        def led_write(value):
            led.value(value if config.LED_ACTIVE_HIGH else not value)

        def buzzer_write(value):
            buzzer.value(value if config.BUZZER_ACTIVE_HIGH else not value)

        return IndicatorController(led_write, buzzer_write)

    def step(self, now=None):
        now = ticks_ms() if now is None else now
        self.pi_service.poll(now)
        self.bus.poll(now)
        self.controller.update(now)
        if (getattr(config, "PI_UNSOLICITED_STATUS_ENABLED", False) and
                self.controller.peer_seen and
                ticks_diff(now, self._last_status_ms) >= config.STATUS_REPORT_MS):
            self.pi_service.send_status(now)
            self._last_status_ms = now

    def run(self):
        print("ESP32 executor started: Pi UART/BLE + actuator UART")
        try:
            while True:
                self.step()
                sleep_ms(config.MAIN_LOOP_MS)
        finally:
            # 未捕获异常或 Ctrl-C 均先向底板发送总停止命令。
            self.drive.stop(emergency=True)
            self.arm.stop()
            self.indicators.cancel_all()
            if hasattr(self.arm, "deinit"):
                self.arm.deinit()
            self.bus.stop(mask=0x03)
            self.bus.poll()
