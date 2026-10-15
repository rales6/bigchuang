# 自然语言任务专用定位：本地实现与接入设计

## 边界及当前状态

本模块只用于后续“去某处、寻找物品、夹取”等任务。没有修改或导入标注巡检状态机；没有连接树莓派、驱动电机或部署服务。`map_camera_inspection.py`、`open_space_panorama.py`、`inspect_map.sh` 均保持本轮开始时内容。

现有地图中粗粒度物体位置可用于搜索区域，不能直接当作车辆停车坐标或强定位地标。目前补充清单没有经过坐标确认的定位锚点，所以实图输入会给出“已确认地标不足”，而不是假装视觉纠错已经可用。照片中看见插座，不等于知道它的准确世界坐标；模型置信度不是定位精度。

## 任务执行定位逻辑

1. 启动时读取原占据地图，确认地图哈希，完成雷达初始定位。标注巡检的初始定位实现继续保留。
2. 任务模式使用独立的位姿管理器：编码器/连续雷达里程计更新位置与朝向，建议 5–10 Hz；UI 同步显示推算位姿、可信位姿和来源。不要用电机指令当作已经发生的运动。
3. 原图雷达匹配负责周期性约束位置；对称区域只保留附近候选，不单帧跳到另一侧。任务目标依照可通行区域和接近点规划，物体中心不能作为车体中心目的地。
4. 照片语义是辅助约束：停车后，约 45 秒/1.5 米、到达目的地、雷达多解、朝向明显不一致时采集。请求冷却 10 秒。无需每个控制帧调用 Qwen，也不为视觉失败原地反复转圈。
5. 语义结果只有通过“已确认固定地标 + 已校准角度 + 新鲜照片 + 多地标一致 + 连续两帧”才能提出修正。默认每次最多平移 0.15 米、修正朝向 8°，不是一次跳变。
6. 视觉失效只放弃视觉修正，继续使用原任务控制与可信雷达/里程计。若主定位也失去可用性，按任务导航自己的传感器/障碍约束减速或停车，不能以“视觉不阻塞”为由无限盲走。该模块不产生电机指令、脱困动作或任务结束指令。

## 照片如何判断/纠正位置

`task_photo_observer.py` 复用 3090 已有 `/agent/ground` 接口，单张照片最多查询三个已确认且类别唯一的固定地标。紧致图像框中心结合相机内参，得到左右方位角；再加该相机相对车体的安装朝向。`CameraCalibration` 的 `x_offset_m/y_offset_m` 表示相机相对底盘中心的前/左安装位移，比较方位时补偿该偏移。双摄必须分别校准，不能认为第二路与第一路方向相同。

`task_semantic_localization.py` 把预测方位角与照片观测角比较。两个不同地标、方位间隔至少 20° 时可以提出朝向修正；三个以上不同地标才允许比较平移候选。平移必须来自雷达支持的附近候选，并由调用者提供完整车体/扫掠范围检查函数。该模块不从单张照片的名称或任意 Qwen 输出计算世界坐标。

同一插座重复检测不能算多个地标。同类别多个柜子/插座且无可靠身份匹配时跳过该类别；只有外部参考图匹配确定身份时，才显式传入 `object_id`。普通 Qwen 响应里的 ID 会被忽略。玻璃外物体、可移动椅子、未确认坐标都不能作为强定位锚点。

候选相对拍照时位姿超过 0.6 米或 65° 不自动接受；单候选朝向细调范围为 45°。例如 180° 朝向翻转会被记录而不瞬间纠正，需任务层停车并重新核对雷达/已知地标。候选之间方位残差差距不足 3° 时判为多解。

照片最长接受 30 秒延迟，但推理期间车辆移动超过 0.1 米/10° 就丢弃结果；必须使用拍照时而不是推理返回时的位姿。重复照片、倒序时间戳不能充当连续确认。定位结果是待应用提案，不会自行更新 SLAM，也不会写原地图。

## 任务层接入接口

```python
from raspberry_pi.task_semantic_localization import (
    TaskSemanticLocalizer, TaskPose, PoseCandidate, TaskPhotoSchedule,
)
from raspberry_pi.task_photo_observer import QwenTaskPhotoObserver

# memory: semantic_object_memory.v1；map_sha256: 实际 PGM 哈希。
# validate_swept_pose: 用旧地图、实际底盘、实时障碍检查整段修正。
localizer = TaskSemanticLocalizer(memory, map_sha256,
                                pose_validator=validate_swept_pose)
schedule = TaskPhotoSchedule()
observer = QwenTaskPhotoObserver(ground_url)

# 后台工作线程；不阻塞高频控制/UI。capture_pose 和时间来自同次采集。
observation = observer.observe(photo_path, camera_index, calibrated_camera,
                               localizer.anchors)
decision = localizer.evaluate(
    frame_id=frame_id, captured_at_s=capture_time, now_s=now,
    capture_pose=capture_pose, current_pose=current_pose,
    detections=observation['detections'], candidates=radar_candidates)
if decision['accepted']:
    # 任务位姿管理器采纳提案，更新其里程计基准；不要修改巡检定位器。
    task_pose_manager.apply_proposal(decision['pose'])
```

这是接入合同，不是已经接通实车的任务调用链。现有 `object_search_mission.py` 仍沿用原逻辑，下一步应在任务模式增加独立开关和位姿管理器后调用以上接口，不能全局替换巡检函数。首次接入先 shadow 模式：仅记候选/残差，不应用修正。

## 不联网的照片结果回放

```powershell
python tools/task_semantic_localization_probe.py --memory semantic_maps/ble_scene_17_20260917_190706/full_object_inventory/semantic_object_memory.json --map backups/semantic_inspection_20260917_195037/pi_snapshot/maps/ble_scene_17.pgm --frames raspberry_pi/examples/task_semantic_frames.json --output simulation_runs/task_semantic_localization/actual_memory_probe.json
python -m unittest tests.test_task_semantic_localization tests.test_task_photo_observer -v
```

示例里的方位观测是合成输入，不是实拍地标。当前实图清单没有已验证锚点，会保持输入位姿。回放 CLI 没有车体验证器，所以不能提出平移；真实任务层必须提供验证器。

## 实车接入前的必要准备

确认至少三个固定地标的准确地图位置、语义类别和参考照片；校准每路相机的内参及相对车体朝向。用语义清单 reviews 文件明确批准可用于定位，保留审核者/时间及地图哈希。建议优先选非对称标牌、不同外观的墙面设施；不要只使用中央桌子或一排相同插座。

离线测试能验证修正方向、角度环绕、重复检测、多解、错误候选、延迟、丢失语义和修正上限；不能证明实际编码器、电机、相机安装或雷达遮挡已正确，也不代表小车已实测跑完整个任务。

本轮扩展回归中，巡检压力测试 `test_real_map_seed_exercises_disturbances_and_finishes_semantics` 有一项 `supplement_count > 0` 断言未通过（实际为 0）。该测试不调用新任务定位模块；未改动对应巡检/仿真实现，保留失败日志，不声明巡检整体通过。
