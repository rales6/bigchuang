# 自然语言任务规划最小闭环

本模块把“帮我处理地上的垃圾”拆为可校验的高层步骤。考虑到 2B 小模型容易漏步骤，
Qwen 只提取意图、目标、目的地、数量和是否需要巡检等紧凑任务契约；确定性编译器再生成
`robot_task_plan.v1`，并校验依赖关系、传感器安全条件和动作范围。输出不会直接驱动车轮
或舵机，需由树莓派端的导航、机械臂适配器逐步确认后执行。

目前支持四类简单任务：

- `cleanup`：搜索垃圾、抓取并送到簸箕或垃圾桶；
- `pick_and_place`：搜索任意可抓物体并送到指定语义地标；
- `navigate`：到桌子、柜子、墙角等标注物附近；
- `inspect`：到语义目标附近观察并报告；

其他能力会编译成 `unsupported`，只报告原因，不产生运动动作。模型给出的标注 id 不仅要
真实存在，还必须与目标查询的名称/类别一致；例如把“蓝色水瓶”错误关联到 `cabinet_01`
会被拒绝，系统转而要求视觉巡检，不会导航到柜子抓取。

## 场景变化与可疑位置层

任务运行过程中发现“物品移动、原位置缺失、出现新物品或外观明显改变”时，写入
`scene_change_overlay.v1`，默认文件为 `outputs/scene_change_overlay.json`。它是原占据地图和
语义标注之上的独立层，原图不会被修改，也不会因为一次不确定识别就删除原标注。

每条可疑位置保存预期物品、地图/栅格坐标、变化类型、置信度和最多 50 条复查历史，并
永久保留 `requires_recheck_on_every_visit=true`。以后自然语言任务涉及该物品或经过相邻
位置时，规划器会把目标设为 `require_scene_recheck=true`；树莓派巡检必须报告对应
`rechecked_suspicious_location_ids`，只看到远处同类物品不足以跳过旧位置复查。

软件可以独立记录和查询变化：

```text
POST /scene-changes/record
GET  /scene-changes
```

记录示例：

```json
{
  "change_type": "moved",
  "expected_item_id": "bottle_01",
  "expected_query": "蓝色水瓶",
  "grid_xy": [12, 10],
  "observed_item_id": "bottle_01",
  "confidence": 0.91,
  "evidence": {"source": "camera", "new_grid_xy": [15, 11]}
}
```

## 本地规则规划与干运行

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_semantic_task_demo.py \
  --instruction "帮我处理地上的垃圾" \
  --backend rules
python scripts/render_demo_scene.py
```

输出：

- `outputs/semantic_task_demo.json`：结构化计划、逐步干运行记录和 A* 路径；
- `outputs/demo_cleanup_scene.png`：占据栅格、语义物品与规划路径。

## 使用 3090 上的 Qwen

```bash
python scripts/run_semantic_task_demo.py \
  --instruction "帮我处理地上的垃圾" \
  --backend qwen \
  --model-path ~/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda
```

如果模型加载、输出 JSON 或安全校验失败，系统会返回同语义的规则计划，并在 `warnings`
中记录回退原因。这样小模型失败不会绕过安全门。

批量评测：

```bash
python scripts/run_task_planner_eval.py \
  --backend qwen \
  --model-path ~/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda \
  --output outputs/task_planner_eval_qwen.json
```

评测覆盖垃圾清理、到工作桌、观察柜子、水瓶搬到工作桌和拒绝唱歌五类指令，并检查
必要/禁止动作以及语义标注 id。2026-08-09 在 3090 上结果为 5/5 通过；模型对水瓶曾输出
语义错误的 `cabinet_01`，安全编译器成功拒绝，评测仍要求最终计划不得接受该 id。

## 启动规划 API

```bash
python scripts/run_semantic_task_planner.py \
  --backend qwen \
  --model-path ~/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda \
  --semantic-map examples/demo_cleanup_scene.json \
  --scene-change-memory outputs/scene_change_overlay.json \
  --host 0.0.0.0 --port 8023
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8023/plan \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"帮我处理地上的垃圾"}'
```

仓库也提供了可直接使用的请求文件：

```bash
curl -X POST http://127.0.0.1:8023/plan \
  -H 'Content-Type: application/json' \
  --data-binary @examples/demo_plan_request.json
```

也可以在请求体的 `semantic_map` 字段直接传入树莓派 `/api/semantic` 的 JSON。若直接使用
现有 `standalone_semantic_map.v2`，规划器会读取其中的 `landmarks`；若要运行栅格干运行，
还需提供 `occupancy_grid.rows`、`robot_pose.grid_xy` 和物体的 `approach_xy`。

## 实机接入边界

后续执行器应逐条消费计划，并把每一步映射到现有能力：

- `patrol_search` / `navigate_to` → 雷达 SLAM 位姿、全局路径规划、视觉/雷达安全仲裁；
- `locate_object` / `locate_destination` → Qwen 首次框选、CSRT 跟踪、视觉-地图融合；
- `align_for_grasp` / `grasp` / `release` → 底盘停止后的机械臂适配器；
- `verify_*` → 摄像头与夹爪反馈；失败时保持 `STOP`。

默认配置只做规划和干运行，不会向真实底盘或机械臂发送命令；必须显式启用两端硬件开关
和同一个临时安全令牌，才允许网关下发已实现的动作。

## 执行状态机与树莓派网关

规划服务现在还提供：

```text
POST /execute/start   装载已生成计划；仿真模式可传 {"auto_run":true}
POST /execute/tick    仅推进一次“下发/确认”循环；硬件模式禁止自动跑完整计划
GET  /execute/state   查询当前步骤、重试次数、安全状态和事件
POST /execute/stop    请求树莓派急停并取消当前任务
```

不传 `--robot-url` 时使用纯仿真桥，所有结果都带
`hardware_command_sent=false`：

```bash
python scripts/run_semantic_task_planner.py \
  --backend rules \
  --semantic-map examples/demo_cleanup_scene.json \
  --host 127.0.0.1 --port 8023

curl -X POST http://127.0.0.1:8023/plan \
  -H 'Content-Type: application/json' \
  --data-binary @examples/demo_plan_request.json
curl -X POST http://127.0.0.1:8023/execute/start \
  -H 'Content-Type: application/json' \
  --data-binary @examples/demo_execute_request.json
```

树莓派参考网关默认也是干运行：

```bash
python scripts/run_pi_task_gateway.py --host 0.0.0.0 --port 8767
```

3090 连接它，但仍不允许真实硬件：

```bash
python scripts/run_semantic_task_planner.py \
  --backend qwen \
  --model-path ~/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --robot-url http://<树莓派IP>:8767 \
  --host 0.0.0.0 --port 8023
```

硬件模式必须在两端显式设置相同的临时安全令牌，并分别添加
`--enable-hardware`。硬件模式禁止 `auto_run`，每个步骤都要由 `/execute/tick`
下发或确认一次：

```bash
export ROBOT_TASK_HARDWARE_TOKEN='现场临时令牌'
python scripts/run_pi_task_gateway.py --enable-hardware --port 8767

export ROBOT_TASK_HARDWARE_TOKEN='同一个现场临时令牌'
python scripts/run_semantic_task_planner.py \
  --backend qwen --robot-url http://<树莓派IP>:8767 \
  --enable-hardware --port 8023
```

配套的树莓派地图版本新增了 `/api/navigation/goal` 与 `/api/navigation/stop`，使用占据栅格
A*、机器人半径膨胀、语义目标安全停靠点、SLAM 稳定性门控和雷达近障停止。因此目前可以
接通 `inspect_semantic_map`、`patrol_search`、`locate_object`、`locate_destination`、
`navigate_to` 和 `report_result`。

机械臂适配层已接到地图主进程独占的 ESP32/BLE 链路，不会另开一个冲突连接。任务网关
只能请求标定文件中预先批准的姿态名，当前固定序列为：

```text
align_for_grasp: pregrasp
grasp:            grasp -> lift -> carry
release:          preplace -> place -> open -> stow
```

它不接受 Qwen 生成的关节编号或舵机脉宽。`robot_arm_pose_profile.v1` 必须包含机器人编号、
标定时间、六关节限位及八个完整姿态；所有脉宽都要位于各关节限位内，且
`calibrated=true`。随代码提供的 `raspberry_pi/arm_profile.template.json` 使用 `null` 脉宽
并设置 `calibrated=false`，因此不能误触发实机。

Stage1 感知候选服务新增 `POST /verify`，只返回目标是否可见/居中、是否在夹爪中、是否
进入目的地和置信度，不返回运动命令。树莓派的 `/api/manipulation/observation` 将当前帧、
雷达前方净空、底盘停止状态与该结果融合；缺帧、超时、低置信度或任一安全门失败都会
同时请求底盘与机械臂停止。

树莓派待部署文件在本仓库的 `raspberry_pi/`；覆盖树莓派现有主程序前应先备份，
并先使用 `--drive-dry-run`。完成实车标定后才可按以下形式启动：

```bash
cd ~/dachuang/car1.0
python -m raspberry_pi.run_pi_semantic_mapper \
  --qwen-url http://<3090-IP>:8011/perceive \
  --drive-control --drive-link ble --drive-protocol vehicle \
  --enable-arm-control \
  --arm-robot-id <当前小车唯一编号> \
  --arm-profile ./raspberry_pi/arm_profile.calibrated.json \
  --manipulation-verify-url http://<3090-IP>:8011/verify
```

规划服务涉及视觉复核时建议保留默认 `--robot-timeout-s 30`；复核期间底盘保持停止。

## 当前验证与剩余现场条件

电脑端新增的变化压力仿真：

```powershell
python -m car_sim.natural_language_task_stress `
  --seed-count 50 `
  --output car_sim/output/natural_language_task_stress_50.json
```

每个随机场景执行到桌子、观察柜子、搬运水瓶、清理垃圾及变化地点复查五类自然语言
任务，并加入八个不写回原图的会话障碍。夹取在仿真中按任务假设默认成功；只有发现本地
抓取模型时才做 smoke test。当前电脑没有完整抓取模型权重，因此模型测试标记为
`skipped_not_installed`，留待树莓派/3090 实机链路验证。

2026-08-09 已完成：

- 3090 上规划、执行状态机、树莓派网关、机械臂姿态门控和 A* 导航共 36 个测试全部通过；
- 垃圾清理 Qwen 计划为 12 步，包含地图巡检、双传感器导航、抓取、投放和视觉复核；
- “把蓝色水瓶放到工作桌上”完成 Qwen → 安全编译 → A* → 抓放符号干运行，最终到达
  `table_01` 的停靠点 `[10, 7]`，没有发送任何硬件命令；
- 规划 API 与干运行树莓派网关的跨 HTTP 闭环已完成。
- 校准就绪的仿真机械臂已完成 12 步垃圾清理链路；撤掉“物体在夹爪中”的视觉证据时，
  网关会阻断任务并同时请求底盘、巡航和机械臂停止。
- 正式 8011 服务已滚动升级并保留旧版备份；`/perceive` 在线回归返回 10 个检测且不含地图坐标，
  新 `/verify` 接口的 mock 正样本和真实 Qwen 负样本均通过。真实 Qwen 在无夹爪证据的地图图像上
  返回 `object_in_gripper=false`，没有把提示词误当成完成证据。

2026-08-11 现场联通进展：

- 树莓派 SSH 已在线；3090 可以 ping 通 `172.18.14.74` 并连接 22 端口。
- 建图服务保持仅监听 `127.0.0.1:8766`，通过树莓派现有的只读代理
  `0.0.0.0:8767` 向 3090 提供 `/api/status`、`/api/semantic` 和 `/api/map.png`，无需暴露控制接口。
- 新的导航、机械臂和语义地图入口已部署到树莓派 `~/dachuang/car1.0/raspberry_pi/`；模块入口
  检查及 10 项无硬件测试通过，没有启动第二个雷达、BLE 或运动进程。
- 本轮建图正常安全停车并保存，最终 PGM、YAML、PNG、轨迹和语义 JSON 已下载到本仓库
  `maps/live_semantic_snapshot_20260811/`。

实车闭环还需要两个现场条件：

1. 当前语义快照只有 2 个 `localized_once` 的 `fixed_structure`，没有确认的垃圾、簸箕、桌子或
   可导航目标。需要布置目标物体并采集多视角观测，或在地图上人工确认它们的名称和坐标。
2. 需要按 `arm_profile.template.json` 实测八个姿态、关节限位和相机到夹爪工作区关系，生成
   独属于当前小车的标定文件；随后用真实“夹住/未夹住、投放成功/失败”图像验证 `/verify`。
   在这些数据通过前，网关会继续阻止真实抓取动作。
