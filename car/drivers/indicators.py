"""非阻塞蜂鸣器和单色状态灯控制。

所有时序都在主循环的 update 中推进，不使用 sleep 或 Timer 中断，因此收到
新的树莓派命令时可以立即替换正在播放的模式。
"""

from core.timebase import ticks_diff, ticks_ms
from protocol.messages import LED_BLINK, LED_OFF, LED_ON


class IndicatorController:
    def __init__(self, led_write, buzzer_write):
        self._led_write = led_write
        self._buzzer_write = buzzer_write
        self.led_mode = LED_OFF
        self.led_period_ms = 0
        self.led_value = False
        self._led_changed_ms = ticks_ms()
        self.buzzer_active = False
        self._beep_repeat_left = 0
        self._beep_on_ms = 0
        self._beep_off_ms = 0
        self._beep_is_on = False
        self._beep_changed_ms = ticks_ms()
        self.cancel_all()

    def set_led(self, mode, period_ms=0, now=None):
        now = ticks_ms() if now is None else now
        if mode not in (LED_OFF, LED_ON, LED_BLINK):
            raise ValueError("unsupported LED mode")
        if mode == LED_BLINK and not 50 <= period_ms <= 10000:
            raise ValueError("LED period must be in range 50..10000")
        self.led_mode = mode
        self.led_period_ms = period_ms
        self._led_changed_ms = now
        self._set_led(mode == LED_ON)

    def beep(self, repeat, on_ms, off_ms, now=None):
        now = ticks_ms() if now is None else now
        if not 1 <= repeat <= 20:
            raise ValueError("beep repeat must be in range 1..20")
        if not 20 <= on_ms <= 5000 or not 20 <= off_ms <= 5000:
            raise ValueError("beep interval must be in range 20..5000")
        self._beep_repeat_left = repeat
        self._beep_on_ms = on_ms
        self._beep_off_ms = off_ms
        self._beep_changed_ms = now
        self.buzzer_active = True
        self._set_buzzer(True)

    def update(self, now=None):
        now = ticks_ms() if now is None else now
        if (self.led_mode == LED_BLINK and
                ticks_diff(now, self._led_changed_ms) >= self.led_period_ms):
            self._set_led(not self.led_value)
            self._led_changed_ms = now

        if not self.buzzer_active:
            return
        interval = self._beep_on_ms if self._beep_is_on else self._beep_off_ms
        if ticks_diff(now, self._beep_changed_ms) < interval:
            return
        self._beep_changed_ms = now
        if self._beep_is_on:
            self._set_buzzer(False)
            self._beep_repeat_left -= 1
            if self._beep_repeat_left <= 0:
                self.buzzer_active = False
        else:
            self._set_buzzer(True)

    def cancel_buzzer(self):
        self.buzzer_active = False
        self._beep_repeat_left = 0
        self._set_buzzer(False)

    def cancel_led(self):
        self.set_led(LED_OFF)

    def cancel_all(self):
        self.cancel_buzzer()
        self.cancel_led()

    def _set_led(self, value):
        self.led_value = bool(value)
        self._led_write(self.led_value)

    def _set_buzzer(self, value):
        self._beep_is_on = bool(value)
        self._buzzer_write(self._beep_is_on)

