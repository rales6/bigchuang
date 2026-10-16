# -*- coding: utf-8 -*-
"""视觉追踪逼近任务的树莓派端集中配置。

所有实车可调参数都集中在这里；命令行参数只覆盖最常用的几项。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    # 2026-08-16 实测：BUFFERSIZE=1 会来回停顿只有 15fps；
    # 2 实测 25fps（相机上限 30），再多会引入陈旧帧延迟。
    buffer_size: int = 2
    # -1 表示自动探测；显式指定用于排障（探测结果会打印在日志里）。
    left_device: int = -1
    right_device: int = -1
    # 2026-09-25 用户现场确认：/dev/video0 是物理右目，
    # /dev/video2 是物理左目。探测顺序是 0/2，拼接前必须交换，
    # 模型和左右掩码对应物理左右 2/0，不再另交换掩码。
    # 再换线后可用 --no-swap-cameras 临时覆盖。
    swap_left_right: bool = True
    # 启动时按档位表实测两路同时帧率，选第一档达标者
    # （1280x720@30 → 1280x720@15 → 960x540@30 → 640x480@30）；
    # 显式指定摄像头编号或改动 width/height/fps 时自动跳过。
    auto_probe: bool = True


@dataclass(frozen=True)
class StereoConfig:
    # 校正后焦距（像素）与基线（米），来自
    # camera_lidar_experiments_20260805_192453/outputs/stereo_calibration.yaml
    # 的 P1/P2 与 baseline_mm_norm（深度废弃后仅供参考）。
    fx_rect_px: float = 1251.497
    baseline_mm: float = 141.107
    # 追踪与瞄准在原始照片空间：正前方目标应落在各相机原始主点
    # K[0][2] 上，不是画面半宽 640。
    # 下列数值按物理左右/示教顺序 video2/video0 存储；追踪若交换
    # 物理左右画面，run_visual_pursuit.geometry_for_camera_order 会同步
    # 交换这两个主点与示教掩码，避免显示顺序改变而停车目标仍用旧顺序。
    cx_raw_left_px: float = 653.81
    cx_raw_right_px: float = 573.41
    # 相机安装 yaw（光学轴相对车头正前方）的像素补偿；
    # 实车首次直行对准验证后再确定符号与数值。
    yaw_compensation_px: float = 0.0


@dataclass(frozen=True)
class PursuitConfig:
    # ---- 3090 服务 ----
    # 8021：8020 已被 arm_grasp_server 占用
    server_ws: str = "ws://192.0.2.20:8021/pursuit_ws"
    server_http: str = "http://192.0.2.20:8021"
    instruction: str = ""
    # 上传帧率上限：0 = 每帧上传（相机帧率，网页显示帧率 = 相机帧率）；
    # 非 0 为可选上限（网络带宽不足时降档，无需改代码）。
    upload_fps: float = 0.0
    jpeg_quality: int = 70
    ground_timeout_s: float = 180.0   # Qwen 跑 CPU 时含模型加载更慢

    # ---- 逼近控制 ----
    stop_distance_m: float = 0.20          # 用户要求：0.2m 左右停下
    stop_confirm_frames: int = 3           # 连续确认帧数
    stop_confirm_margin_m: float = 0.03    # 距离确认容差（视差噪声 ~2cm 量级）
    # 视觉优先实车档：限制巡航速度，30fps 下单帧位移更小，给全局
    # 重捕获留出余量；静摩擦由下方短起步冲量和卡死升级链处理。
    max_linear_mm_s: int = 180
    # 2026-08-17 用户指定 160：冷态静摩擦时有时无（车跑热后
    # 低档也能动）。160 推不动时靠 stuck 升级链加冲量破摩擦。
    # 注意：min 必须 < max！三个速度斜坡（示教/行测距/距离）都用
    # max-(max-min)×ratio 公式，min==max 时斜坡全部退化为恒速 280，
    # 30px 示教容差窗口 <1 帧就被穿过，3 帧确认永远凑不满——
    # 示教停车从未达成的直接原因（2026-08-17 修复）。
    min_linear_mm_s: int = 120
    # 静止起步逐渐加速（用户要求）：
    # 起步瞬间给 linear_breakaway_mm_s 冲量破静摩擦，保持 hold 秒后
    # 回落到目标速度按 linear_accel_mm_s2 爬升。2026-08-17 实测
    # 冷态破静摩擦需要 ~460mm/s 冲量（雷达距离不变直到第 5 级
    # 460 才动）→ stuck 升级链 280→340→400→460→520 覆盖。
    # ESP32 固件上限 MAX_LINEAR_MM_S=550、COMMAND_MAX_TTL_MS=2500。
    linear_accel_mm_s2: float = 250.0
    linear_decel_mm_s2: float = 400.0
    # 不再用 480mm/s 瞬时冲量：实车会弹射起步。改为仅
    # 略高于巡航速度的温和起步，静摩擦由长脉冲时间解决。
    linear_breakaway_mm_s: int = 220
    linear_breakaway_hold_s: float = 0.10
    # 雷达自适应脉冲：每个整圈 ICP 窗口若确认底盘没动，
    # 下一脉冲只延长时间，不提高速度。连续确认已移动后才
    # 缩短一档，避免在静摩擦临界点来回振荡。
    stuck_window_s: float = 1.0          # 雷达 ICP 移动观测窗口
    stuck_move_eps_m: float = 0.015      # 小于 1.5cm 才确认没动，留出 ICP 噪声
    stuck_progress_confirm_m: float = 0.040  # 至少 4cm 才确认已移动并回退脉冲档位
    stuck_yaw_eps_rad: float = 0.02
    stuck_escalate_windows: int = 1      # 一次确认没动 → 下一脉冲立即延长
    stuck_breakaway_step_mm_s: int = 0   # 卡住只延长通电时间，不再加大瞬时速度
    stuck_max_escalations: int = 14      # 达最大时长后仍留多次尝试再停车
    # SEARCHING 旋转卡死保护（2026-08-16 实测转了 45s 无进展）：
    # 连续该数量的 stall 窗口没有航向进展 → 跳过本步转下一步。
    search_stall_giveup_windows: int = 6
    full_speed_distance_m: float = 1.2     # 大于该距离用巡航速度
    slow_start_distance_m: float = 0.8     # 小于该距离开始线性减速
    max_angular_mrad_s: int = 140          # 追踪转向限幅，防目标甩出画面
    max_angular_near_mrad_s: int = 80      # 近端进一步限制摆头
    # 2026-08-16 用户要求：目标偏左就略微向左前进（边前进边修正），
    # 不是原地对准。横向偏差超过 deadband 后按比例降低前进速度：
    # 偏差每增大 speed_scale_px 速度再降一档，最低保留 min_ratio。
    advance_align_deadband_px: float = 60.0
    advance_speed_scale_px: float = 300.0
    # 最低保留 50%：25% 只有 40mm/s，实测托不住滚动反而触发卡死升级
    advance_min_ratio: float = 0.5
    angle_gain_mrad_per_px: float = 1.0    # 对车身参考线容差之外的偏差温和转向
    # 近距（<near_ttl_distance_m）增益比例：近距大目标框中心抖动
    # 绝对值大，全增益 2.5 mrad/px 让转向来回猛甩（2026-08-17 实测
    # 近距 w 在 ±250mrad 振荡）。
    angle_gain_near_ratio: float = 0.5
    angle_deadband_px: float = 12.0        # 越出参考容差后仍保留小死区，防止抖动
    # 2026-09-26 停车现场：用户确认地缝与车身中线共线。
    # 固定两路的投影，不随小车转向重新检测地缝。坐标按各路宽高归一化；
    # 640x480 预览左(197,121)->(457,479)，右(404,82)->(113,479)。
    forward_reference_enabled: bool = True
    forward_left_line: tuple = ((197/640,121/480),(457/640,479/480))
    forward_right_line: tuple = ((404/640,82/480),(113/640,479/480))
    # 单独实拍核对1280x720：其视场与640x480不同，禁止直接等比套用。
    forward_left_line_1280: tuple = ((454/1280,190/720),(850/1280,719/720))
    forward_right_line_1280: tuple = ((765/1280,128/720),(333/1280,719/720))
    forward_reference_band_ratio: float = 0.035  # 每侧约22px@640，留出检测抖动
    # 2026-08-17 安全：相邻帧 bbox 中心跳变超过该像素视为锁漂移，
    # 立即停车等重新确认（防对漂移框全速冲撞）。
    bbox_jump_px: float = 120.0
    # 感知结果超过该年龄立即把直线速度置零；旧逻辑在 3090 卡顿后会
    # 最多 5s 持续复用最后一框，低帧率近距冲出视野的主要控制风险。
    perception_max_age_s: float = 0.45
    # 连续控制时，两次有效视觉观测之间最多允许盲走约 45mm；低于
    # 电机有效速度时由下方脉冲伺服接管，不命令不可实现的低速。
    max_blind_travel_mm: float = 45.0
    ttl_far_ms: int = 600                  # 远端运动指令 TTL
    ttl_near_ms: int = 400                 # 近端（<0.5m）运动指令 TTL
    near_ttl_distance_m: float = 0.5

    # ---- 近距蠕动逼近（2026-08-17 修复近距卡死）----
    # 雷达安全限速（<0.6m 限 50mm/s）低于电机静摩擦临界（热态
    # ~160mm/s），低速巡航推不动 → stuck 升级链反复触发、车在
    # 0.34-0.54m 反复蠕动到 GIVE_UP（实测两轮从未达成 0.2m 停车）。
    # 近距改为脉冲式：短促冲量（>摩擦临界）→ 完全停止 → 测量，循环，
    # 每脉冲前进 ~5-8cm，停止期雷达/视觉测量稳定，停车判定可靠。
    # 1.0m = 覆盖整个雷达限速区（<1.0m 限 100mm/s，再被横向 align
    # 压到 50 会低于摩擦临界推不动——实测第十轮在 0.84m 卡死 GIVE_UP）。
    creep_zone_m: float = 1.00       # 低于该距离进入蠕动模式
    creep_pulse_mm_s: int = 405      # 实车验证刚好能稳定越过静摩擦
    creep_pulse_s: float = 0.80
    creep_pulse_near_s: float = 0.80
    creep_pulse_step_s: float = 0.10 # 雷达确认没动，下一脉冲 +0.10s
    creep_pulse_max_s: float = 2.00  # 速度不变，只逐级增加通电时间
    creep_rest_s: float = 0.45
    # 不依赖雷达的视觉近距伺服：进入示教慢速带/行测距近带后，改用
    # 短脉冲前进→完全停车→等待新视觉结果。硬件只能可靠执行较高
    # 起步速度时，用时间占空比获得厘米级平均步长。
    # 此底盘固件还会把速度按 650mm/s² 斜坡输出：旧 0.07~0.14s
    # 脉冲结束时内部速度只有 45~90mm/s，根本越不过静摩擦。改用稍长
    # 的可自适应脉冲，并在 rest 阶段由上层发送 MSG_STOP 硬停；因此
    # 不需要持续高速，也不会在目标丢失后继续滑行。
    pulse_drive_enabled: bool = True
    visual_pulse_mm_s: int = 405
    visual_pulse_far_s: float = 0.80
    visual_pulse_near_s: float = 0.80
    visual_pulse_step_s: float = 0.10
    visual_pulse_max_s: float = 2.00
    visual_pulse_rest_s: float = 0.45
    visual_feedback_step_s: float = 0.05
    visual_feedback_max_s: float = 0.95
    visual_feedback_max_levels: int = 3
    lost_stationary_recheck_s: float = 1.0
    lost_stationary_recheck_limit: int = 2
    lost_stationary_recheck_timeout_s: float = 20.0
    reground_dual_confirm_frames: int = 2
    search_pulse_s: float = 0.80
    search_pulse_rest_s: float = 0.50
    search_total_on_time_s: float = 2.40

    # ---- 丢失后倒车找回 ----
    # 单目仍可见：根据可见相机和框在画面中的位置，倒车小弧线
    # 扩大视野；双目都丢失：根据丢失前数帧的平均水平位置与
    # 运动趋势选择回退方向。每步后急停观测，恢复双目才继续前进。
    recovery_enabled: bool = True
    recovery_history_frames: int = 6
    recovery_prediction_horizon_s: float = 0.45
    recovery_direction_deadband_px: float = 80.0
    recovery_settle_s: float = 0.25
    recovery_max_s: float = 4.5
    recovery_reverse_mm_s: int = 405
    recovery_turn_mrad_s: int = 320
    recovery_reverse_pulse_s: float = 0.32
    recovery_rest_s: float = 0.38
    recovery_rear_half_width_deg: float = 28.0
    recovery_rear_clear_m: float = 0.32
    recovery_dual_confirm_frames: int = 2
    recovery_max_reverse_pulses: int = 3
    recovery_scan_max_age_s: float = 0.5
    # 现场确认后向扫描被车身遮挡：不再读取/解释后向回波。
    # 此时只允许有近期前进证据的受限原路短退，不能无限盲退。
    recovery_rear_sensor_usable: bool = False
    # 后向雷达的固定近距回波可能来自底盘本体。仅对刚刚走过且视觉
    # 证实在逼近目标的路径，小步回退并每步停车重新观测；固件从静止
    # 按 650mm/s² 爬升，两次短脉冲理论总位移 <= 10cm。
    recovery_swept_path_enabled: bool = True
    recovery_swept_path_max_age_s: float = 1.6
    recovery_swept_path_pulse_s: float = 0.40
    recovery_swept_path_max_m: float = 0.10
    recovery_swept_path_max_pulses: int = 2
    recovery_swept_path_accel_mm_s2: float = 650.0
    recovery_swept_path_turn_mrad_s: int = 160
    # A recent forward arc that sweeps the binocular target across the image
    # may be undone along its just-traversed path before a slow Qwen recheck.
    recovery_turn_rollback_max_age_s: float = 2.5
    recovery_turn_rollback_jump_px: float = 220.0
    recovery_turn_rollback_edge_px: float = 250.0
    recovery_self_echo_max_m: float = 0.13

    # ---- 视觉转向脉冲 ----
    # 四轮差速底盘原地转向需要克服很大的轮胎侧向摩擦。实车即使
    # (0,1600) 脉冲持续 80s 也几乎不转，因此改成带前进牵引的弧线
    # 转弯：轮胎先滚动，再由左右轮差速改变航向。
    # 内八双目常有 30-70px 的单眼边界偏差。此前 30px 就触发
    # 405mm/s、0.8s 的弧线，实车日志里目标接近时因此被甩出视野。
    # 保留大偏差主动转向，小偏差继续直行并在下一帧重测。
    align_turn_trigger_px: float = 80.0
    # 实车验证 260mm/s 的弧线脉冲无法稳定越过轮胎/地面静摩擦；405
    # 与直行、蠕动脉冲使用同一个已验证起步档位，低平均速度仍由短脉冲
    # 与完整停车观测实现，而不是命令硬件无法执行的低瞬时速度。
    align_arc_linear_mm_s: int = 405
    align_turn_pulse_mrad_s: int = 450
    align_turn_pulse_s: float = 0.80
    align_turn_pulse_step_s: float = 0.10
    align_turn_pulse_max_s: float = 1.20
    # Grab-service image space (640x480), not physical metres.
    # Toe-in physical left/right; image Y grows as a floor target approaches.
    # These are image rows, NOT metres or 4D hull-distance thresholds.
    hull_micro_enter_left_y_px: float = 300.0
    hull_micro_enter_right_y_px: float = 220.0
    hull_micro_exit_row_margin_px: float = 20.0
    hull_micro_step_s: float = 0.40
    # Every normal forward pulse retains the prior test's sixth window floor;
    # a weak response must not reset the next request to .40s.
    # Reverse, yaw and low-confidence probes retain the shorter base window.
    hull_forward_first_s: float = 0.60
    # Normal approach arc: operator confirmed .40s has only slight traction.
    # Do not change close pure-yaw, reverse or low-confidence probe windows.
    hull_approach_arc_first_s: float = 0.60
    hull_micro_forward_patience_steps: int = 2
    hull_grasp_margin_px: float = 25.0  # each independent 640x480 view's 2D hull
    hull_micro_forward_step_worsening_px: float = 8.0
    hull_micro_forward_total_worsening_px: float = 12.0
    hull_multicue_progress_enabled: bool = True
    hull_center_jitter_px: float = 6.0
    hull_area_growth_ratio: float = 1.03
    hull_forward_trend_max_age_s: float = 30.0
    hull_micro_speed_mm_s: int = 415  # only near-field micro; pursuit/approach stay 405
    hull_micro_step_max_s: float = 0.68
    hull_micro_step_increment_s: float = 0.04
    hull_micro_max_steps: int = 16
    # Raw 1280x720 rows: hand off BEFORE the lower blind edge, regardless
    # of mask X error. Grab preview has a separate 640x480 calibration.
    visual_align_handoff_enabled: bool = True
    visual_align_handoff_left_row_px: float = 400.0
    visual_align_handoff_right_row_px: float = 320.0
    visual_align_near_pulse_max_s: float = 0.64
    retreat_recent_max_angular_mrad_s: int = 600
    retreat_recent_max_heading_rad: float = 0.55
    # 640x480 body-line error, after the existing soft band. Small heading
    # errors should not interrupt approach with four-wheel skid steering.
    hull_approach_turn_trigger_px: float = 40.0
    # Retained for old consumers; hull adjustment no longer escalates stationary
    # yaw to this setpoint. It replans after two failures instead.
    hull_micro_turn_retry_mrad_s: int = 900
    hull_micro_turn_static_steps: int = 2
    align_turn_rest_s: float = 0.45

    # ---- 距离来源 ----
    # 2026-08-16 用户移动过摄像头，双目标定失效：不用双目深度
    # （服务端已加 --no-depth 对应开关）。
    use_stereo_depth: bool = False
    # 雷达扫描高度也照不到低矮地面目标（用户确认），改用"视觉行测距"：
    # 目标中心在画面中越低说明越近。追踪已改在原始照片空间，
    # 行阈值直接按原始照片行算：下 1/3（480）开始减速，
    # 下 1/10（648，用户可接受区间 1/10~1/3）停车。
    # 实车微调这两个值即可改变停车距离。
    use_visual_row_distance: bool = True
    visual_near_row_px: float = 400.0    # 2026-08-17 提前减速：0.7m 近距目标尺度剧变导致丢锁
    visual_stop_row_px: float = 648.0
    # 即使左右掩码/单目锁异常，近端大目标也不能再发下一段前进脉冲。
    # 这是防冲入双目盲区的安全兜底，触发时停为 give_up，不宣称可抓取。
    visual_blind_guard_row_px: float = 500.0
    visual_blind_guard_box_height_px: float = 120.0

    # ---- 示教停车（2026-08-16 用户要求，优先于行测距）----
    # 目标中心对齐左/右目掩码图中的示教位置时停车。掩码图
    # grab_visual/left_graspable.png 与 right_graspable.png（640x480）
    # 质心 ×2 得 1280x720 坐标：左 (772,545)、右 (526,421)；
    # 与 3090 上 arm_teach_artifacts/model_state.json 的 input_means
    # ×(1280,720) 左(765,547) 右(485,421) 基本一致（右目差 41px，
    # 以掩码图为准）。这些坐标按模型物理左右 video2/video0 顺序存放；
    # 若追踪拼接顺序交换，运行时必须同步交换。停车应按整个白色掩码区域，而不是只按质心附近
    # 30px：下列多边形是 left_picture/right_picture 的白区轮廓从
    # 640x480 缩放到相机原图 1280x720 后的近似凸轮廓。
    use_taught_stop: bool = True
    taught_require_both_views: bool = True
    physical_left_on_left: bool = True  # display may be reversed; yaw must follow physical eye
    taught_left_cx_px: float = 772.0
    taught_left_cy_px: float = 545.0
    taught_right_cx_px: float = 526.0
    taught_right_cy_px: float = 421.0
    # 不在刚进入掩码时就停；左右目中心都需比各自掩码
    # 质心更靠画面下方该像素数，即实车终点再向目标靠近一些。
    taught_closer_than_centroid_px: float = 30.0
    taught_left_mask_polygon_px: tuple = (
        (422.0, 478.0), (378.0, 549.0), (464.0, 609.0),
        (650.0, 630.0), (1018.0, 630.0), (1112.0, 596.0),
        (1160.0, 507.0), (1142.0, 492.0), (870.0, 454.0),
    )
    taught_right_mask_polygon_px: tuple = (
        (138.0, 370.0), (142.0, 422.0), (180.0, 452.0),
        (278.0, 488.0), (564.0, 524.0), (948.0, 464.0),
        (968.0, 444.0), (964.0, 429.0), (736.0, 346.0),
        (322.0, 327.0),
    )
    taught_mask_margin_px: float = 18.0
    # 正常小步把预留从50收紧到30px；参考刚完成的一步视觉变化。
    # 没有可信近期反馈/单步过大时仍用50px保守缓冲，不能忽略曾经
    # 再发0.8s就跳出画面的实测风险。不把像素当成厘米。
    taught_early_stop_margin_px: float = 30.0
    taught_early_stop_uncertain_margin_px: float = 50.0
    taught_stop_feedback_max_age_s: float = 5.0
    taught_stop_normal_step_max_px: float = 40.0
    # 保留旧字段供外部配置兼容；新停车判断使用完整掩码多边形。
    taught_stop_tolerance_px: float = 30.0
    taught_slow_band_px: float = 220.0   # 2026-08-17 加宽慢速带：近距提前减速防丢锁
    # 抓取掩码附近绝不继续增加单次脉冲时长。2026-09-25 实测：
    # 0.8~0.9s 小步前进，最后 stuck 等级把脉冲加到 1.0s 后突然
    # 破静摩擦，双目目标中心单步跳 109/96px，实车仍冲过头。
    taught_close_pulse_band_px: float = 350.0
    taught_close_pulse_max_s: float = 0.80
    taught_fine_pulse_band_px: float = 120.0
    taught_stationary_trial_max_s: float = 0.85

    # ---- 雷达辅助 ----
    lidar_front_half_width_deg: float = 21.0   # 前向安全扇区半宽
    lidar_hard_floor_m: float = 0.12           # 任何时刻前方硬停止距离（只做安全兜底）
    lidar_min_range_m: float = 0.12
    lidar_max_range_m: float = 8.0

    # ---- 丢失后旋转重搜 ----
    # 先完全停车，让服务端全画面重识别有时间追回；短时丢框期间不再
    # 按最后方位继续转动车身。超过该时间才进入主动旋转搜索。
    lost_before_search_s: float = 1.8
    # 相对丢目标时航向的确定性旋转序列（度，正=左转）；转满仍未找到即放弃。
    # 2026-08-16 由近及远：目标多半在附近，先 ±25° 小角度搜再逐步扩大。
    search_rotation_steps_deg: tuple = (25.0, -50.0, 90.0, -140.0, 180.0,
                                        -25.0, 0.0)
    # 2026-08-16 增强：整个序列不再相对丢目标时的航向，而是以
    # "最后看见目标的方位"为中心展开（known-target re-acquisition：
    # 先把车头回瞄目标最后方位，再交替外扩，找回概率显著更高）。
    # 最后方位 = 丢失瞬间最后 bbox 中心相对瞄准点的偏角 atan(err/f)，
    # 钳位到 ±search_first_aim_back_max_deg。
    ego_focal_px: float = 600.0   # 像素↔角度换算焦距（≈标定 K[0][2] 量级）
    search_first_aim_back_max_deg: float = 60.0
    # 旋转重搜中服务端视觉追回（模板/YOLO 提案）连续确认该帧数 →
    # 立即停车并提交 Qwen 复检，不必转完本步——避免转过头把目标
    # 又转出视野（2026-08-16 视觉先行找回）。
    search_reacquire_confirm_frames: int = 3
    # 搜索转满仍没找到时，GIVE_UP 停车后再自动提交一次 Qwen 复检
    # （目标可能在最后一步旋转中才进入画面），找到即恢复追踪。
    search_turn_speed_mrad_s: int = 900    # 实测 450/750 无法克服四轮侧向摩擦
    search_turn_start_kick_mrad_s: int = 1600  # 短起步增益，之后回落到扫视速度
    search_turn_ttl_ms: int = 500
    search_turn_tolerance_deg: float = 5.0
    search_step_obstacle_clear_m: float = 0.30  # 目标扇区需要此距离无障碍才转过去
    search_stall_s: float = 1.5                 # 该时间内航向无进展视为卡死→加起步增益
    # 每步旋转最大时长（2026-08-17 实测：ICP yaw 估计在旋转中来回跳
    # （-23°→2°→-9°），卡死检测被"伪进展"骗过，step 4 卡了 15s+。
    # 超过时限直接跳过本步，不依赖 yaw 判定）。
    search_step_timeout_s: float = 12.0
    search_icp_max_correspondence_m: float = 0.35
    # ICP 连续该数量的卡死窗口从未收敛过一次 → 按时长估算转角
    # （空旷/对称环境特征不足时防止实车无限原地旋转）。
    search_icp_fallback_windows: int = 4
    search_time_fallback_efficiency: float = 0.5  # 时长估算的转角效率（保守）

    # ---- ESP32 链路 ----
    link_mode: str = "ble"
    ble_device_name: str = "ESP32-Robot-Car"
    # 固定地址避免名称发现受附近同名设备或 BlueZ 名称缓存影响。
    ble_address: str = "AA:BB:CC:DD:EE:01"
    # ESP32 从旧连接断开后重新开始广播偶尔需要超过默认 6 秒。
    ble_connect_timeout_s: float = 10.0
    # ESP32 已支持 balanced_twist 的逐轮输出倍率（允许 0.5~1.5）。
    # 静摩擦问题的主因是旧脉冲太短，还没走完 ESP32 内部加速斜坡
    # 就急停，而非缺少 1.5× 的瞬时推力。1.5× 实车会弹射起步；
    # 撤掉所有额外倍率，靠 0.70~1.20s 的温和长脉冲越过静摩擦。
    wheel_output_gain: float = 1.0
    dry_run: bool = True                        # 默认只打印，不驱动实车
    log_csv: str = "pursuit_log.csv"


@dataclass(frozen=True)
class RunConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    pursuit: PursuitConfig = field(default_factory=PursuitConfig)
