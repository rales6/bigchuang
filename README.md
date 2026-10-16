# 语义导航与双目夹取小车（公开源码版）

自然语言指令 → 任务规划 → 区域导航与观察 → 双目识别、接近与微调 → 模型夹取 → 搬运与地面放置。

本仓库只公开源码、通用文档和离线测试。**真实房间地图、照片、雷达录制、设备标定、内部审计清单及真实配置不随 Git 分发**，由项目成员通过内部渠道交接。旧提交曾包含内部资料；本次常规清理提交不等于已经清除 Git 历史。详见[公开范围与内部交接](docs/publication.md)。

## 获取代码

```bash
git clone https://github.com/rales6/bigchuang.git robot-car
cd robot-car
```

清理版合并到 `main` 前，使用拉取请求所指的 `privacy/20261016` 分支查看。不要继续从旧交接分支分发内部资料。[下一组入口](docs/handoff_github.md)说明了需要单独取得的运行依赖。

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
| 内部地图的本地放置约定 | `maps/current/`，公开版仅有说明 |

详见[功能索引](docs/features.md)与[组件边界](docs/architecture.md)。不同运行根内的同名驱动不能随意合并，开发镜像和正式服务代码修改后需要核对一致。

## 电脑离线快速开始

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe tools/project.py paths
.\.venv\Scripts\python.exe tools/project.py check
.\.venv\Scripts\python.exe -m tools.maintenance.public_hygiene
.\.venv\Scripts\python.exe tools/project.py test --suite smoke
.\.venv\Scripts\python.exe tools/project.py test --suite regression
```

Linux/macOS 使用 `.venv/bin/python`。这些命令不连接设备、不打开真实摄像头、不驱动电机/机械臂。依赖真实录制或地图的测试会明确跳过；它们不是已经通过的实车验证。其他离线套件及依赖见[依赖说明](requirements/README.md)。

## 网页与设备端启动

必须先取得团队的内部地图/标定/模型包，按设备实际布局配置服务路径、网络、相机与权限。代码中的 `192.0.2.*`、`198.51.100.*`、`.example.invalid` 及 `/home/robot`、`/home/gpu` 仅为示例，**不是现用设备地址**。不配置这些依赖，公开源码不能直接完成实机任务。

设备管理员已正确安装并配置服务后，Pi终端可使用：

```bash
sudo systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service
systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service
curl -fsS http://127.0.0.1:8780/health
```

3090终端先进入自己的追踪服务根目录，再执行：

```bash
bash final_work/run_3090_background.sh
curl -fsS http://127.0.0.1:8030/api/health
```

不要在同一端重复启动服务。确认机器人在线后，用设备管理员提供的网页地址发送任务，例如“去柜子边夹取一个绿色小物块，然后放在距离最近的椅子旁”。本次上传没有部署这些改动，也没有启动小车。

## 地图、模型、配置与发布

- `.env.example` 仅是空模板，不会被应用自动加载。真实密钥、密码、硬件令牌和数据库不得提交；旧密钥是否撤销由账户持有人操作。
- 地图、参考照片、场景雷达录制、玻璃边界、标定和模型权重通过内部渠道提供，保留设备上已验证版本。不要用公开示例覆盖现用配置。
- 速度学习状态属于设备运行数据，留在设备，不用 Git 覆盖或复位。
- 维护工具连接设备前要求显式配置主机及独立核验的 SSH 主机密钥指纹；不能把真实指纹或地址硬编码回公开仓库。
- 发布前检查 Git 暂存内容；`.gitignore` 不会自动清除已经跟踪的文件或旧历史。

```bash
git status --short
git diff --stat
git add -A
python -m tools.maintenance.public_hygiene --staged
git diff --cached --stat
```

确认范围后提交到功能分支，通过拉取请求审阅。不要强推覆盖团队历史，也不要把内部交接备份目录整体上传。
