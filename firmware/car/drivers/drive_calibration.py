"""底盘左右轮速度相关补偿及掉电保存。"""

try:
    import ujson as json
except ImportError:
    import json

try:
    import uos as os
except ImportError:
    import os


class DriveCalibration:
    VERSION = 1
    MAX_ABS_TRIM = 0.15
    MAX_ABS_SLOPE = 0.0005

    def __init__(self, trim_intercept=0.0, trim_slope_per_mm_s=0.0, path=None):
        self.path = path
        self.trim_intercept = 0.0
        self.trim_slope_per_mm_s = 0.0
        self.set(trim_intercept, trim_slope_per_mm_s, persist=False)

    @classmethod
    def load(cls, path):
        calibration = cls(path=path)
        loaded = False
        for candidate in (path, path + ".bak"):
            try:
                with open(candidate, "r") as stream:
                    data = json.load(stream)
                if int(data.get("version", 0)) != cls.VERSION:
                    raise ValueError("unsupported calibration version")
                calibration.set(
                    data["trim_intercept"],
                    data["trim_slope_per_mm_s"],
                    persist=False,
                )
                loaded = True
                break
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if not loaded:
            # 文件缺失或损坏时安全退回无补偿，不阻止底盘启动。
            calibration.reset(persist=False)
        return calibration

    def set(self, trim_intercept, trim_slope_per_mm_s, persist=True):
        intercept = float(trim_intercept)
        slope = float(trim_slope_per_mm_s)
        self._validate(intercept, slope)
        self.trim_intercept = intercept
        self.trim_slope_per_mm_s = slope
        if persist:
            self.save()

    def reset(self, persist=True):
        self.trim_intercept = 0.0
        self.trim_slope_per_mm_s = 0.0
        if persist:
            self.save()

    def trim_for_speed(self, speed_mm_s):
        trim = (
            self.trim_intercept
            + self.trim_slope_per_mm_s * abs(float(speed_mm_s))
        )
        return _clamp(trim, -self.MAX_ABS_TRIM, self.MAX_ABS_TRIM)

    def values(self):
        return self.trim_intercept, self.trim_slope_per_mm_s

    def save(self):
        if not self.path:
            return
        temporary = self.path + ".tmp"
        backup = self.path + ".bak"
        data = {
            "version": self.VERSION,
            "trim_intercept": self.trim_intercept,
            "trim_slope_per_mm_s": self.trim_slope_per_mm_s,
        }
        with open(temporary, "w") as stream:
            json.dump(data, stream)
        _remove_if_present(backup)
        had_previous = False
        try:
            os.rename(self.path, backup)
            had_previous = True
        except OSError:
            pass
        try:
            os.rename(temporary, self.path)
        except Exception:
            if had_previous:
                os.rename(backup, self.path)
            raise
        _remove_if_present(backup)

    @classmethod
    def _validate(cls, intercept, slope):
        if not -cls.MAX_ABS_TRIM <= intercept <= cls.MAX_ABS_TRIM:
            raise ValueError("trim intercept outside safe range")
        if not -cls.MAX_ABS_SLOPE <= slope <= cls.MAX_ABS_SLOPE:
            raise ValueError("trim slope outside safe range")
        for speed in (0.0, 550.0):
            trim = intercept + slope * speed
            if not -cls.MAX_ABS_TRIM <= trim <= cls.MAX_ABS_TRIM:
                raise ValueError("calibration exceeds safe trim range")


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def _remove_if_present(path):
    try:
        os.remove(path)
    except OSError:
        pass
