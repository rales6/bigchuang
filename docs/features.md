# 功能交接索引

## 建图

`robot/raspberry_pi/autonomous_mapping.py` 为采集/建图入口；`mapping/slam.py`、`scan_matcher.py`、`occupancy_grid.py`、`scan_filter.py` 负责扫描匹配、地图更新和过滤，`lidar/n10_driver.py` 负责原始雷达。

## 语义标注与地图制作

`map_camera_inspection.py` 获取照片及拍摄方向；`services/perception/scripts/run_inspection_annotation.py` 等负责模型标注。`robot/tools/build_relationship_semantic_map.py` 融合多批结果与人工复核布局，保留其脚本，不把原始照片当最终位置真值。`build_semantic_release.py` 生成发布包；当前结果在 `maps/current/`。

## 根据指令到工作区域

`robot/grasp_range_integration/phase1/deepseek_planner.py` 解析物品与地点；`visibility_route_planner.py` 规划有效观察范围；`relationship_find_only_entry.py` 接入地图/雷达导航。恢复、车身回波、玻璃边界分别在同目录模块，不因本次整理改变门限。

普通区域导航由 `services/track/final_work/pi/region_navigation.py` 接入 `region_localization.py`；携带物品时使用 `carry_localization.py`。先核验新鲜雷达与视觉证据，稳定时衔接导航，证据不足时进入恢复，不用目标坐标冒充小车位置。具体场景地图、标定和部署审计属于内部资料，公开版仅保留算法与入口。

2026-10-15 第二轮部署补齐目标与转向衔接：区域筛选过滤 `behind_glass=true`、`search_region_allowed=false` 及家具上的标记，保留颜色条件，不能用蓝柜替换黄柜。未标颜色的唯一室内候选只作为待视觉复核区域，不修改稳定地图的标注事实。区域站位失败改抛现有恢复器可识别的 `NoReachableVisibilityError`，优先使用原退距，必要时扩大到 1.50/1.70m。`phase1/adaptive_turn.py` 按左右方向保留最近实测有效的起转速度、时长及轮输出档位；仍按剩余角度减小脉冲，保留实时净空/玻璃边界、反向纠偏和反馈复核。

## 识别、接近、微调与夹取

`services/track/pursuit/run_visual_pursuit.py`、`pursuit_controller.py`、`hull_micro_adjust.py` 执行视觉追踪和最后微调。`services/grasp/gpu/arm_grasp_server.py` 与 `arm_teach_model.py` 输出舵机预测；`services/grasp/pi/arm_teach.py` 执行动作。

当前微调设置以同步回来的 `services/grasp/pi/start_web.sh` 和 `grasp_pose_correction.py` 为准，不按旧对话或历史 README 猜测。`phase=done` 是动作序列完成，不是独立的夹持力/抓牢确认。

## 搬运、放置和录像

`services/track/final_work/pi/robot_skills.py` 保存技能交接；`region_navigation.py` 选择/前往放置区域；`placement_extension.py` 执行地面下降—松爪—抬臂。只支持已实现的地面放置，不推断桌面或椅面的高度。

2026-10-15 新增原地放置：“放下”“原地放下小物块”“把手里的物品放到地上”编译为 `place_on_floor → verify_completion → report_result`，目标带 `selection=in_place`、`allow_chassis_motion=false`。跳过语义地点选择、雷达定位、导航和到达记录；仍需本次携带记录、抓取落地姿态、空闲底盘以及现有机械臂互斥/急停与 BLE 校验。地点放置流程不变，放置超时仍查询原任务编号而非重复松爪。编译器与执行器的维护副本同步更新，相关无硬件测试位于 `tests/test_in_place_placement.py`。本地修改需要另行部署、重载后才能在设备服务中使用。

`task_video.py`、`camera_broker.py` 记录和共享双目视频；运行文件保留在设备 `final_work/task_videos/latest/`，不进入仓库。

## 工具、测试与仿真

总仓库工具统一从 `python tools/project.py` 开始。老 `deploy_*`、`test_*`、`probe_*` 工具可能连接设备或启动动作，不能因为名字像测试就纳入默认离线测试。默认只收集顶层 `tests/`，额外组件回归由显式套件清单选择。
