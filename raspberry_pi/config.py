"""树莓派程序的集中配置。

这里的默认值适合先做桌面联调。上车前应通过命令行参数或修改本文件确认串口、
雷达方向和地图尺寸。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SerialConfig:
    port: str = "/dev/serial0"
    baudrate: int = 230400
    timeout_s: float = 0.03
    request_timeout_s: float = 0.45
    retries: int = 2
    keepalive_period_s: float = 0.25
    # uart: 仅串口；ble: 仅蓝牙；auto: UART 超时后自动切换 BLE。
    link_mode: str = "auto"
    ble_device_name: str = "ESP32-Robot-Car"
    ble_address: Optional[str] = None
    ble_connect_timeout_s: float = 8.0
    # A GATT operation must fail quickly; using the discovery timeout here
    # previously allowed one stuck write to block motion for 8-12 seconds.
    ble_operation_timeout_s: float = 1.5
    # Per-wheel output multipliers are owned by the Raspberry Pi. Channel
    # order: front-left, front-right, rear-left, rear-right.
    wheel_output_gains: tuple = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class LidarConfig:
    port: str = "/dev/ttyUSB0"
    # 镭神 N10 串口版固定使用 230400, 8N1。
    baudrate: int = 230400
    min_distance_m: float = 0.12
    max_distance_m: float = 8.0
    angle_offset_deg: float = 0.0
    # N10 俯视角度顺时针递增；建图坐标系要求逆时针为正。
    clockwise: bool = True
    motor_control: bool = True
    min_scan_points: int = 100


@dataclass(frozen=True)
class MappingConfig:
    resolution_m: float = 0.05
    width_cells: int = 400
    height_cells: int = 400
    max_scan_points: int = 420
    max_correspondence_m: float = 0.35
    min_match_points: int = 35
    max_match_rmse_m: float = 0.16
    min_match_inlier_ratio: float = 0.38
    coarse_yaw_range_deg: float = 35.0
    coarse_yaw_step_deg: float = 4.0
    icp_trim_fraction: float = 0.85
    outlier_neighbor_distance_m: float = 0.45
    # Range-adaptive point clusters remove isolated rays while preserving
    # farther walls, whose adjacent laser beams are naturally farther apart.
    outlier_cluster_base_radius_m: float = 0.06
    outlier_cluster_radius_per_meter: float = 0.025
    outlier_cluster_max_radius_m: float = 0.28
    outlier_cluster_min_neighbors: int = 2
    # Anchor frame-to-frame ICP against a short, recent local map.
    local_submap_scan_count: int = 8
    local_submap_max_points: int = 1000
    local_submap_voxel_m: float = 0.07
    local_submap_max_rmse_m: float = 0.14
    local_submap_min_inlier_ratio: float = 0.42
    # Sparse long-term keyframes provide a global anchor in addition to the
    # rolling six-scan local map.
    global_keyframe_translation_m: float = 0.18
    global_keyframe_rotation_deg: float = 10.0
    global_map_max_points: int = 3200
    global_map_voxel_m: float = 0.08
    global_match_every_updates: int = 3
    global_match_max_correction_m: float = 0.18
    global_match_max_correction_deg: float = 10.0
    relocalization_min_keyframes: int = 4
    relocalization_max_correction_m: float = 0.22
    relocalization_max_correction_deg: float = 12.0
    relocalization_max_rmse_m: float = 0.10
    relocalization_min_inlier_ratio: float = 0.55
    lidar_offset_x_m: float = 0.0
    lidar_offset_y_m: float = 0.0
    map_keyframe_translation_m: float = 0.025
    map_keyframe_rotation_deg: float = 2.0
    map_keyframe_max_interval_s: float = 0.50
    map_min_linear_speed_m_s: float = 0.025
    map_max_linear_speed_m_s: float = 0.30
    map_max_angular_speed_rad_s: float = 0.65
    map_speed_filter_window: int = 3
    mapping_max_distance_m: float = 3.0
    # Revisited mapping keyframes replace older evidence around the same pose.
    # Replaying these keyframes lets old free space return to unknown and lets
    # a newly observed wall replace an older position/orientation.
    # A 15-minute run can produce about 1,800 mapping keyframes at the
    # 0.50-second gate. Keep the full session instead of dropping the oldest
    # rooms after only a few minutes.
    map_rebuild_max_keyframes: int = 1800
    map_revisit_replace_radius_m: float = 0.30
    map_revisit_replace_yaw_deg: float = 25.0
    # Keep several independent observations from the same viewpoint. One new
    # scan cannot erase an older wall; only a sustained run of newer evidence
    # gradually retires the oldest keyframes in that direction.
    map_revisit_evidence_frames: int = 8
    map_contradiction_clear_hits: int = 15
    map_auto_expand: bool = True
    map_expand_margin_m: float = 1.0
    render_wall_gap_max_m: float = 0.15
    render_wall_support_m: float = 0.10
    min_obstacle_area_m2: float = 0.03
    max_pose_linear_speed_m_s: float = 0.55
    pose_translation_margin_m: float = 0.025
    max_pose_translation_step_m: float = 0.12
    max_pose_angular_speed_rad_s: float = 4.5
    pose_rotation_margin_deg: float = 8.0
    # Even when scan timestamps are delayed, one ICP update may not rotate
    # the map by an implausibly large amount.
    max_pose_rotation_step_deg: float = 15.0
    # The real chassis slips along a short arc while turning. Classify the
    # motion from lidar and accept plausible translation instead of assuming
    # a perfect wheel-commanded spin.
    lidar_turn_min_rotation_deg: float = 1.5
    lidar_turn_max_translation_m: float = 0.08
    pure_rotation_max_translation_m: float = 0.08
    manhattan_enabled: bool = True
    manhattan_min_segments: int = 28
    manhattan_min_confidence: float = 0.62
    manhattan_anchor_observations: int = 8
    manhattan_max_error_deg: float = 8.0
    manhattan_correction_gain: float = 0.18
    manhattan_max_correction_deg: float = 0.8
