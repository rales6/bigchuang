# 语义导航与双目夹取小车

自然语言指令 → 任务规划 → 区域导航与观察 → 双目识别、接近与微调 → 模型夹取 → 搬运与地面放置。

项目集成 Android 指令交互、ESP32 底盘与机械臂控制，以及运行在树莓派和 GPU 服务器上的建图、感知与任务执行服务。支持按指定地点和物品组织搜索、夹取、搬运与地面放置任务。

## 获取代码

```bash
git clone https://github.com/rales6/bigchuang.git robot-car
cd robot-car
```

首次使用请先阅读[部署准备](docs/handoff_github.md)和[配置说明](docs/configuration.md)。

## 按功能找代码

| 功能 | 目录或入口 |
| --- | --- |
| 雷达建图与扫描匹配 | `robot/raspberry_pi/autonomous_mapping.py`、`mapping/`、`lidar/` |
| 拍照、语义标注、多组地图融合 | `robot/raspberry_pi/map_camera_inspection.py`、`robot/tools/`、`services/perception/scripts/` |
| 指令规划、观察视野、导航恢复 | `robot/grasp_range_integration/phase1/` |
| Qwen / YOLO 感知 | `services/perception/`、`services/track/gpu/` |
| 视觉接近、后退与凸包微调 | `services/track/pursuit/` |
| 机械臂执行 / 模型服务 | `services/grasp/pi/`、`services/grasp/gpu/` |
| 网页、技能编排、搬运放置、双目录像 | `services/track/final_work/` |
| 手机端 / 固件 / CAD | `apps/android/`、`firmware/car/`、`hardware/cad/` |
| 仿真 / 工具 / 测试 | `simulation/`、`tools/`、`tests/` |
| 场景地图配置目录 | `maps/current/`，数据按部署环境准备 |

详见[功能索引](docs/features.md)与[组件边界](docs/architecture.md)。不同运行根内的同名驱动不能随意合并，开发镜像和正式服务代码修改后需要核对一致。

## 电脑离线快速开始

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe tools/project.py paths
.\.venv\Scripts\python.exe tools/project.py check
.\.venv\Scripts\python.exe tools/project.py test --suite smoke
.\.venv\Scripts\python.exe tools/project.py test --suite regression
```

Linux/macOS 使用 `.venv/bin/python`。以上为离线检查，不连接硬件；需要场景数据的用例在数据未配置时会跳过。其他套件及运行依赖见[依赖说明](requirements/README.md)。

## 网页与设备端启动

先按[配置说明](docs/configuration.md)准备地图、模型、相机标定和服务路径，设置各设备的网络与权限。文档中的地址和目录是示例，请替换为自己的部署配置。

服务安装配置完成后，在树莓派终端执行：

```bash
sudo systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service
systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service
curl -fsS http://127.0.0.1:8780/health
```

在 GPU 服务器终端进入追踪服务根目录，再执行：

```bash
bash final_work/run_3090_background.sh
curl -fsS http://127.0.0.1:8030/api/health
```

确认服务健康、机器人在线后，打开部署的网页地址，输入任务，例如：

```text
去柜子边夹取一个绿色小物块，然后放在距离最近的椅子旁
```

实际运行前确认相机顺序、机械臂状态和通道条件，现场保留急停方式。避免在同一设备上重复启动服务。

## 进一步阅读

- [功能索引](docs/features.md)
- [组件边界](docs/architecture.md)
- [配置说明](docs/configuration.md)
- [维护与发布](docs/maintenance/README.md)
- [协作约定](CONTRIBUTING.md)
