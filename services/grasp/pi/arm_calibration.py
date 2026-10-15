"""机械臂-摄像头视觉标定参数。

机械臂参数（来自 car/config.py）：
  Joint 0 (base yaw):   500-2500µs, home=1500µs, 10µs/deg
  Joint 1 (shoulder):   800-1700µs
  Joint 2 (lift):       1500-2200µs
  Joint 3 (elbow):      800-1500µs
  Joint 4 (wrist):      900-2100µs
  Joint 5 (claw):       1100-1600µs, open=1200µs, closed=1600µs

笛卡尔空间 (calibrated_delta 模式):
  forward_mm: 220-300 (home=250)
  up_mm:      50-130  (home=80)
  left_deg:   ±80     (home=0)
  claw:       0-100   (0=闭合, 100=全开)

摄像头:
  位置:   x=0.15m, y=±0.07m, z=0.095m
  yaw:    约 +4.43° (略微偏右)
  分辨率: 1280×720 (可从配置文件获得)
  立体:   LEFT|RIGHT side-by-side

标定映射（需实物实验标定，以下为初始估计值）:
  - 基座旋转 1° → 画面像素位移约 8-15px（取决于爪子在画面中的深度）
  - 升降 1mm   → 画面像素位移约 1-3px
  - 伸 1mm     → 爪子面积约增加 0.3-0.7%
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ArmVisualCalibration:
    """像素-机械臂运动映射参数。

    所有像素坐标系基于：摄像头拍摄的单半侧画面 (640×720 或 1280×720)。
    如果使用 side-by-side 立体图 (2560×720)，则需要先转为单侧坐标。
    """

    # ── 机械臂物理参数 ────────────────────────────────────
    arm_forward_range_mm: tuple[float, float] = (220.0, 300.0)
    arm_up_range_mm: tuple[float, float] = (50.0, 130.0)
    arm_yaw_range_deg: tuple[float, float] = (-80.0, 80.0)
    arm_claw_range: tuple[int, int] = (0, 100)
    arm_home_forward_mm: float = 250.0
    arm_home_up_mm: float = 80.0
    arm_home_yaw_deg: float = 0.0

    # ── 像素映射参数 ──────────────────────────────────────
    # 画面中爪子水平位移 1px → 基座旋转角 (deg)
    # 近处大、远处小；默认取中等距离的近似值
    deg_per_pixel: float = 0.08

    # 画面中爪子垂直位移 1px → 升降高度 (mm)
    mm_per_pixel_vertical: float = 0.50

    # ── 深度估计参数 ──────────────────────────────────────
    # 爪子完全伸出时在画面中占据的面积占比（相对整个画面）
    claw_max_area_ratio: float = 0.15
    # 每前进 1mm 爪子面积增加的相对比例
    area_gain_per_forward_mm: float = 0.008

    # ── 对齐与抓取阈值 ────────────────────────────────────
    # 水平对齐容差 (px)
    horizontal_alignment_px: float = 18.0
    # 垂直对齐容差 (px)
    vertical_alignment_px: float = 15.0
    # 爪子/目标面积比 > 此值时认为可抓取
    grasp_area_ratio_threshold: float = 0.50
    # 最小面积比 (目标太小则拒抓)
    min_target_area_ratio: float = 0.005

    # ── 控制步长限制 ──────────────────────────────────────
    max_left_per_step_deg: float = 12.0
    max_up_per_step_mm: float = 15.0
    max_forward_per_step_mm: float = 25.0
    min_step_duration_ms: int = 500

    # ── 爪子参数 ──────────────────────────────────────────
    claw_open_value: int = 95
    claw_closed_value: int = 10
    claw_partial_close_value: int = 40

    # ── 安全限制 ──────────────────────────────────────────
    max_attempts_per_step: int = 8
    max_total_attempts: int = 40
    reverse_on_overshoot: bool = True

    # ── 摄像头参数 ────────────────────────────────────────
    # 画面尺寸（半侧单相机，非 side-by-side）
    camera_width: int = 640
    camera_height: int = 480

    # ── 爪子颜色特征（HSV 范围，用于辅助跟踪）─────────────
    claw_color_hsv_low: tuple[int, int, int] = (0, 50, 80)
    claw_color_hsv_high: tuple[int, int, int] = (25, 255, 255)
    claw_color_enabled: bool = False

    def save(self, path: str | Path) -> None:
        data = {k: v for k, v in self.__dict__.items()}
        # 转换 tuple 以便 JSON 序列化
        for k, v in data.items():
            if isinstance(v, tuple):
                data[k] = list(v)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ArmVisualCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # tuple 字段还原
        tuple_fields = {
            "arm_forward_range_mm", "arm_up_range_mm",
            "arm_yaw_range_deg", "arm_claw_range",
        }
        for k in tuple_fields:
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        if "claw_color_hsv_low" in data and isinstance(data["claw_color_hsv_low"], list):
            data["claw_color_hsv_low"] = tuple(data["claw_color_hsv_low"])
        if "claw_color_hsv_high" in data and isinstance(data["claw_color_hsv_high"], list):
            data["claw_color_hsv_high"] = tuple(data["claw_color_hsv_high"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ── 便捷方法 ──────────────────────────────────────────

    def clamp_forward(self, value: float) -> float:
        return max(self.arm_forward_range_mm[0],
                   min(self.arm_forward_range_mm[1], value))

    def clamp_up(self, value: float) -> float:
        return max(self.arm_up_range_mm[0],
                   min(self.arm_up_range_mm[1], value))

    def clamp_yaw(self, value: float) -> float:
        return max(self.arm_yaw_range_deg[0],
                   min(self.arm_yaw_range_deg[1], value))

    def clamp_claw(self, value: int) -> int:
        return int(max(self.arm_claw_range[0],
                       min(self.arm_claw_range[1], value)))

    def step_duration(self, forward_mm: int, up_mm: int, left_deg: int) -> int:
        """根据移动量估算运动时长 (ms)，匹配 _cartesian_duration 逻辑。"""
        linear = max(abs(int(forward_mm)), abs(int(up_mm)))
        angular_equiv = abs(int(left_deg)) * 3
        distance = max(linear, angular_equiv)
        duration = int(max(700 + distance * 12, 900, self.min_step_duration_ms))
        return min(duration, 3200)


# 默认标定实例，方便直接使用
DEFAULT_CALIBRATION = ArmVisualCalibration()
