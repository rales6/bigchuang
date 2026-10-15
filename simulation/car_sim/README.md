# CAR·LAB 二维仿真样品

## 开阔点全景巡检闭环仿真

优先执行正式树莓派状态机（硬件、时间和扫描里程计使用扰动模型）：

```powershell
python -m car_sim.open_space_panorama_runtime_stress --profile realistic --seed-count 5 --output simulation_runs/open_space_panorama_runtime.json
```

检查真实几何覆盖、碰撞、照片朝向误差和图片编号唯一性，前后校验原图哈希。
这是控制流程仿真，不是实际 ICP 或 Qwen 准确率测试。

`open_space_panorama_stress.py` 使用原始 `ble_scene_17` 地图，复用新模式的雷达方向选择、
站点净空判定和照片几何覆盖统计。没有固定目标点或预先提供的全局路径；只在局部安全
方向中偏向未覆盖区域。四轮、玻璃、雷达、地图边缘和指令扰动来自现有压力仿真场景。
地图纠错低频触发，注入的远处对称候选不能直接接管位姿。

```powershell
python -m car_sim.open_space_panorama_stress `
  --profile realistic `
  --seed-count 20 `
  --output car_sim/output/open_space_panorama_realistic_20.json
```

真实相机识别仍需实车验证；仿真的 90% 指标是旧地图边界进入照片视场的几何覆盖率。

## 自然语言任务与场景变化压力仿真

`natural_language_task_stress.py` 会在随机会话障碍和移动物品条件下反复执行导航、观察、
水瓶搬运、垃圾清理和可疑位置复查。夹取在电脑仿真中默认成功；本地存在抓取模型时只做
可用性 smoke test，不存在则记录 `skipped_not_installed`。变化写入独立 overlay，原地图
摘要在每轮前后做 SHA-256 校验，任何覆盖原图都会使测试失败。

```powershell
python -m car_sim.natural_language_task_stress `
  --seed-count 50 `
  --output car_sim/output/natural_language_task_stress_50.json
```

## 已知地图语义巡检压力仿真（推荐）

`semantic_inspection_stress.py` 是无需网页、树莓派、BLE、相机、GPU 或网络服务的
系统级闭环仿真。它直接加载 `maps/ble_scene_17.pgm/.yaml`，复用项目现有的候选
拍摄点生成器和路径控制函数，并在隐藏的“实际场景”中加入：

- 随位置变化的静摩擦和低摩擦斑块；
- 四个轮子各自不同的增益与启动阈值；
- 单轮连续多帧空转（编码器仍可能报告运动）；
- 电池输出下降、BLE/TTL 指令在电机爬升完成前过期；
- 轮组受载、启动扭矩不足，以及短 TTL 在底盘完成加速前过期；
- 保存地图与实际墙体边缘的厘米级差异；
- 左侧玻璃墙的雷达穿透、无效外部回波；
- 雷达角度/距离噪声、量化、丢点、近距离散斑和多径长回波；
- 指令延迟、直行偏航以及实际运动方向与输入不完全一致；
- 单帧跳变拒绝和连续两帧一致候选的温和重锚；
- 视觉 Top-3 粗定位、短距离移动后的第二帧融合和全图定位 fallback；
- 全部规划点拍照、双相机一致性确认和整块黑色组件语义提升。
- 对可见墙面每 30 cm 采样，并放置不改变雷达占据栅格的标牌、开关、海报等视觉目标；
- 自动补充被拐角遮挡的少量广角拍摄位，验证墙面和“雷达不可见”物品的实际相机覆盖；
- 前后 360° 障碍余量、路径横向偏差、连续空转、底盘墙距和碰撞的逐种子硬验收。
- 前方雷达净空充足时自动放宽启动脉冲；只有障碍进入完整停车包络才按碰撞处理。
- 每完成一个点后重新为剩余目标规划，同时计入出发转向和抵达后拍摄转向成本。
- 必要时在机位前增加对准接近点，避免抵达后再原地掉头；并硬性检查直行弧线的单步大幅偏航。
- 模拟拍摄全部结束和停车后再批量进行语义标注，不让在线识别占用巡检运动时间。

在项目根目录运行标准压力回归：

```powershell
python -m car_sim.semantic_inspection_stress `
  --profile realistic `
  --seed-count 20 `
  --output car_sim/output/semantic_inspection_realistic_20.json
```

进一步放大四轮差异、空转、地图边缘变化、玻璃穿透和雷达错误：

```powershell
python -m car_sim.semantic_inspection_stress `
  --profile harsh `
  --seed-count 20 `
  --output car_sim/output/semantic_inspection_harsh_20.json
```

退出码为 0 只在以下条件同时成立时出现：每个随机种子都完成全部拍摄点、全部可规划
永久黑色组件都获得严格语义确认、视觉 Top-K 包含正确地点盆地、定位 P95 不超过
13 cm、连续大定位误差不超过 2 帧、横向路径偏差不超过 40 cm、零碰撞、零未恢复的目标超时、
连续有指令但不运动不超过 8 帧，墙面覆盖不低于 90%、雷达不可见墙面物品覆盖不低于
95%，并且上述每类扰动确实在批次中被触发。报告旁会生成一个代表种子的：

- `*_artifacts/semantic_map.json`：白色自由区、灰色未知区和已确认黑色组件语义；
- `*_artifacts/semantic_map.png`：语义着色、实际轨迹、相机视场、墙面覆盖点和拍摄位置；
- `*_artifacts/captures/*.jpg`：电脑生成的模拟拍照证据。

三档建议用途：`mild` 用于快速冒烟，`realistic` 作为日常回归，`harsh` 用于验证
控制与定位不会因连续空转或玻璃异常锁死。仿真标签是场景真值模拟，不代表真实相机
模型的识别精度；真实视觉精度仍需用实际参考照片做离线交叉验证。

该目录是一个完全独立的本地仿真器，不会导入或修改项目其他目录。它提供：

- 可调整尺寸的矩形房间；
- “任务选择 / 场景布置 / 仿真建图 / 机械臂视角 / 脚本与指令”五页工作区；
- 强制“选择任务 → 布置场景 → 锁定场景并运行”的分步流程；
- 场景编辑器使用“房间 / 布置 / 小车 / 偏差 / 控制”按钮分组，并可一键隐藏或打开；
- 矩形、圆形障碍物的绘制、选择、拖动和删除；
- 可放置、拖动、选择尺寸和删除的夹取物品，以及车头前置摄像头画面；
- 可通过角度输入、旋转按钮或画布拖拽设置小车初始朝向；
- 可配置长宽的小车、差速运动与碰撞保护；
- 180°/360°可配激光扫描和实时占据栅格建图；
- 左右轮倍率、打滑、控制延迟、速度扰动、角速度偏置、雷达测距噪声和丢点；
- `set_twist`、`goto`、`set_arm_joints`、`arm_stop` 等外部指令；
- 与现有 `Esp32Client`、`N10LidarDriver` 常用接口兼容的虚拟硬件；
- 模仿 OpenCV `VideoCapture.read()` 的 `SimulatedCameraClient`；
- 目标点 A* 路径规划、指令日志和状态回传。

## 启动

在项目根目录运行：

```powershell
python -m pip install -r requirements-pi.txt
python car_sim/server.py
```

然后打开 <http://127.0.0.1:8765>。

## 从其他 Python 文件发指令

```python
from car_sim.simulator_client import SimulatorClient

sim = SimulatorClient()
sim.set_twist(250, 0, 600)
sim.stop()
sim.goto(6.4, 3.8)
print(sim.state())
```

也可以直接请求 HTTP：

```text
POST http://127.0.0.1:8765/api/commands
Content-Type: application/json

{"type":"set_twist","linear_mm_s":250,"angular_mrad_s":0,"ttl_ms":600}
```

正线速度表示向车头前进，正角速度表示逆时针左转，与项目的 Vehicle Link V2
坐标约定一致。线速度限制为 `-550..550 mm/s`，角速度限制为
`-3500..3500 mrad/s`，与当前小车配置一致。`set_twist` 的 TTL 到期后会自动停车；
新速度覆盖旧速度，不排队。

## 推荐的程序接入方式

不要抓取另一个脚本的 `print()` 输出。输出文本没有稳定格式，也不能把雷达帧可靠地
送回建图程序。推荐让算法依赖“硬件接口”，测试时注入虚拟实现，上车时注入真实实现：

```python
from car_sim.virtual_hardware import (
    SimulatedEsp32Client,
    SimulatedLidarDriver,
)

car = SimulatedEsp32Client()
lidar = SimulatedLidarDriver()

car.start()
try:
    for scan in lidar.scans():
        # scan 是项目现有的 LaserScan：
        # angles_rad、distances_m、timestamp_s
        update = slam.process(scan)
        car.set_twist(180, 0, 450)
finally:
    car.stop()
    lidar.close()
    car.close()
```

如果原脚本已经直接创建 `Esp32Client` 和 `N10LidarDriver`，可通过运行器临时替换，
无需修改原文件：

```powershell
python -m car_sim.run_with_simulator raspberry_pi/mapping_app.py `
  --output car_sim/output/my_map --max-scans 200
```

运行器只在当前 Python 进程内替换硬件类，不会修改 `raspberry_pi` 或 `car`。

## 双向接口

| 方向 | 接口 | 用途 |
|---|---|---|
| Python → 网页 | `POST /api/commands` | 速度、停车、导航和复位指令 |
| 网页 → Python | `GET /api/lidar?after=N` | 长轮询下一帧雷达点 |
| 网页 → Python | `GET /api/state` | 实际速度、位姿、关节位置和摄像头检测结果 |
| 网页 → Python | `GET /api/scene` | 房间、障碍物和待夹取物品真值，仅供自动评测 |

雷达扫描使用车体坐标系：x 向车头、y 向车左、角度逆时针为正，与项目约定一致。
建图算法不应读取 `ground_truth_pose`；它只供测试阶段计算定位误差。

## 如何测试物品夹取

在“任务选择”中选择“物品识别与夹取”，通过“场景布置 → 布置 → 放置物品”
设置目标，然后运行：

```powershell
python -m car_sim.pickup_demo
```

脚本会导航到物品前方，网页自动切换到摄像头和机械臂视角，再依次发送项目真实的
`set_arm_joints()` 指令：0号关节左右对准、1–3号关节协同下降、4号腕部旋转、
5号夹爪闭合，最后抬起物品。

自定义视觉程序可以直接读取仿真摄像头：

```python
from car_sim.virtual_hardware import SimulatedCameraClient

camera = SimulatedCameraClient()
ok, frame = camera.read()       # 360×640 BGR numpy 图像
detections = camera.detections()  # bbox、距离、水平偏角和置信度
```

机械臂脉宽限制与 `car/config.py` 一致，超出任一关节安全范围的指令会被仿真服务拒绝。

## 如何判断建图功能是否实现

先打开网页并保留默认场景及“轻微真实偏差”，然后执行：

```powershell
python -m car_sim.mapping_benchmark --scans 160
```

如需与自己的控制脚本使用完全相同的速度：

```powershell
python -m car_sim.mapping_benchmark --scans 160 --speed 350 --turn-speed 2500
```

脚本会控制车辆走短边矩形路线，把网页雷达帧送入项目现有 `LidarSlam`，最后将结果写入
`car_sim/output/`。测试结束后会自动生成一份可直接打开的 HTML 分析报告，并从五个方面
给出 PASS/FAIL：

1. 扫描匹配接受率不低于 55%；
2. 地图融合帧数达到最低要求；
3. 已观测栅格数量足够；
4. 地图中确实形成占用栅格；
5. 平均扫描匹配 RMSE 不高于 0.16 m。

例如输出前缀为 `car_sim/output/my_mapping_test` 时，会生成：

- `my_mapping_test_report.html`：总览、指标判定、PNG 地图、RMSE 趋势和分析建议；
- `my_mapping_test_metrics.json`：结构化实验参数、汇总指标、阈值和全部文件绝对路径；
- `my_mapping_test_scans.csv`：每一帧的位姿、RMSE、匹配点数、速度和拒绝原因；
- `my_mapping_test.png`：带估计轨迹的建图结果；
- `my_mapping_test.pgm`、`.yaml` 和 `_trajectory.csv`：地图及轨迹原始文件。

如仅用于不需要报告的自动化流水线，可增加 `--no-report`。

建议按三档依次测试：

1. **理想模型**：应该稳定通过，用来验证程序和坐标约定；
2. **轻微真实偏差**：应该仍能通过，用来验证基础鲁棒性；
3. **明显偏差/压力测试**：允许指标下降，但程序不能崩溃、地图不能完全发散。

最终还应打开生成的 PNG/PGM，与网页房间结构人工核对：墙和障碍物方向正确、没有明显
重影，闭合路线回到起点附近。只生成图片不代表建图成功，必须同时检查定位连续性、
匹配接受率、地图覆盖率和几何一致性。
