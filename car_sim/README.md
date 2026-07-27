# CAR·LAB 二维仿真样品

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
