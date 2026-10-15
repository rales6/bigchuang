# 语义导航与双目夹取小车

Android / 网页指令 → DeepSeek 任务规划 → 雷达与语义地图导航 → 双目识别、靠近、微调 → 模型夹取 → 搬运与地面放置。

这是整合后的**电脑本地总仓库**。保留原 `car1.0` 的 Git 历史，收回树莓派和 3090 的在用源码；历史实验、录像、旧地图和虚拟环境不混入源码。此次整理不部署远端、不启动车辆，软件测试也不等于实车验收。

2026-10-15 后续源码同步、到达判定调整和 DeepSeek 密钥清理位置，见[交接清单](docs/maintenance/20261015_handoff_and_deepseek.md)。密钥仅列位置，交接前由原成员自行清空并在供应商侧撤销；不要将真实配置加入Git。

## 按功能找代码

| 功能 | 主要代码 |
| --- | --- |
| 建图、雷达、扫描匹配 | `robot/raspberry_pi/mapping/`、`autonomous_mapping.py`、`lidar/` |
| 拍照与多批语义标注 | `robot/raspberry_pi/map_camera_inspection.py`、`robot/tools/build_relationship_semantic_map.py`、`services/perception/scripts/` |
| 自然语言解析、观察区域规划、导航恢复 | `robot/grasp_range_integration/phase1/` |
| Qwen / YOLO 识别 | `services/perception/src/`、`services/track/gpu/` |
| 视觉接近、后退、凸包微调 | `services/track/pursuit/` |
| 树莓派机械臂服务 | `services/grasp/pi/` |
| 3090 夹取模型服务、训练与标定 | `services/grasp/gpu/` |
| 网页与技能编排、双目录像 | `services/track/final_work/` |
| Android 手机端 | `apps/android/` |
| ESP32 底盘与舵机执行 | `firmware/car/` |
| CAD | `hardware/cad/` |
| 仿真 | `simulation/car_sim/`，不包含旧运行输出 |
| 工具与测试 | `tools/`、`tests/`，各组件原有测试保留在组件内部 |
| 当前在用地图及语义资料 | `maps/current/` |

核心 Python 包、固件协议和设备上的绝对路径没有为了美化目录强行改名。`robot/` 保留 car1 运行边界，`services/track/` 保留视觉服务的另一套运行边界；两者原本就是不同部署目录，不能把同名驱动随意覆盖成一份。详见 [架构与目录](docs/architecture.md)、[功能入口](docs/features.md)。

## 快速开始

### 1. 网页实机测试：启动已有部署

以下 Linux 命令针对**设备上已经部署、配置好的版本**，不是在电脑新仓库中直接执行。目录整理没有自动部署到设备；电脑源码更新也不会自动让运行中的服务生效。

电脑 PowerShell 中登录树莓派（IP 变化时，在树莓派运行 `hostname -I` 查询）：

```powershell
ssh dachuang@172.18.12.242
```

在**树莓派终端**启动网页桥和到 3090 的反向隧道，检查连接：

```bash
sudo systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service
systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service
curl -fsS http://127.0.0.1:8780/health
```

两个服务应为 `active`，健康接口应返回 `ok: true`。需要开机自动启动时执行一次：

```bash
sudo systemctl enable final-work-pi-bridge.service final-work-pi-tunnel.service
```

使用 systemd 管理时，不要同时运行 `run_pi_background.sh` 或另起一份隧道脚本，避免重复进程占用端口。上述服务启动本身不下发车辆运动或机械臂动作。

再开一个**电脑 PowerShell 窗口**登录 3090：

```powershell
ssh -p 2020 sjtu@202.121.181.124
```

在 **3090 终端**启动网页并检查端到端连接：

```bash
cd /home/sjtu/track
bash final_work/run_3090_background.sh
curl -fsS http://127.0.0.1:8030/api/health
curl -fsS http://127.0.0.1:18780/health
```

启动脚本会检查已有网页进程，不必反复启动；仍需以健康接口结果确认就绪。`18780` 是 3090 上经隧道访问树莓派桥的入口，不是网页端口。

电脑浏览器打开：[网页测试入口](http://192.168.55.33:8030/)。这是当前部署的局域网地址，需要电脑能访问该网络；电脑上的 `127.0.0.1:8030` 默认并不是此服务。

网页中可输入：

```text
去柜子边夹取一个绿色小物块，然后放在距离最近的椅子旁
```

已有本次夹取记录、想直接在当前位置放到地面时，可输入（已于 2026-10-15 部署并重载两端服务）：

```text
原地放下小物块
```

也支持“放下”“把手里的物品放到地上”。该流程不要求地点、不定位、不启动底盘导航，而是核对持物记录与 BLE/双摄后，复用本次夹取落地姿态下降、松爪、抬臂并留照。若没有有效的本次夹取姿态，不会猜测关节位置或直接高处松爪；指定“放在椅子旁”等地点仍走搬运导航。

更新后请重新发送指令生成新计划，不要直接继续更新前失败的旧任务。旧任务的步骤不会因服务重载自动改写。部署备份及验证范围见 [本轮部署记录](docs/maintenance/20261015_runtime_deployment.md)。

先查看规划，再按网页提示确认执行。**确认执行会让小车转向、前进、夹取和搬运**；请确认起点、夹爪状态、物品和通道已准备好，并在现场看护、可急停。仅打开网页或检查健康接口不会启动任务。

### 2. 双目预览、模型检查和日志

没有正在运行的实车任务时，可在**树莓派终端**打开双目预览并检查 BLE、相机状态（开启摄像头和连接，不下发运动）：

```bash
curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8780/actions/preview/start
curl -fsS http://127.0.0.1:8080/api/status
```

检查 `ble` 是否为 `true`，左右摄像头 `frames` 是否持续增加、`age_s`（最新帧距今秒数）是否保持较小。预览不等于任务录像；正式任务的视频由任务流程开启。任务进行中不要手动重启桥服务、蓝牙或相机。

在 **3090 终端**只读检查识别、夹取和视觉跟随服务（这些命令不会加载新模型或启动任务）：

```bash
curl -fsS http://127.0.0.1:8010/health
curl -fsS http://127.0.0.1:8020/health
curl -fsS http://127.0.0.1:8021/health
```

这些模型服务应已按现有部署常驻。若未启动，按对应组件文档排查；不要重复加载大模型或直接用其他启动脚本替换正在运行的服务。

**树莓派日志：**

```bash
journalctl -u final-work-pi-bridge.service -n 80 --no-pager
journalctl -u final-work-pi-tunnel.service -n 80 --no-pager
tail -n 80 /home/dachuang/dachuang/track/final_work/logs/camera_preview.log
```

**3090 网页日志：**

```bash
tail -n 80 /home/sjtu/track/final_work/logs/agent_web.log
```

若网页能打开但机器人离线，先检查树莓派供电、IP 和两个 systemd 服务，再检查 3090 的 `18780/health`；不要为了恢复网页连接直接启动寻物脚本。

### 3. 电脑端安装与离线检查

在**电脑 PowerShell**中执行，不需要连接树莓派，也不会打开摄像头、雷达或电机：

```powershell
cd D:\xingxianfeng_files\dachuang\robot-car
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe tools/project.py paths
.\.venv\Scripts\python.exe tools/project.py check
.\.venv\Scripts\python.exe tools/project.py test --suite smoke
.\.venv\Scripts\python.exe tools/project.py test --suite regression
```

已有 `.venv` 时可跳过创建和安装。Linux/macOS 将解释器替换为 `.venv/bin/python`，并改成自己的仓库路径。

其他离线测试按需执行：

```powershell
.\.venv\Scripts\python.exe tools/project.py test --suite firmware
.\.venv\Scripts\python.exe tools/project.py test --suite annotation
.\.venv\Scripts\python.exe tools/project.py test --suite simulation
# 一次执行全部离线测试
.\.venv\Scripts\python.exe tools/project.py test --suite all-offline
```

`simulation` 是仿真回归，不是实车启动命令。GPU、树莓派和 Android 的运行依赖分开安装，见 [依赖说明](requirements/README.md)。

### 4. Android 调试构建（可选）

电脑已配置 Android SDK 和项目所需 JDK 后，在 PowerShell 执行：

```powershell
cd D:\xingxianfeng_files\dachuang\robot-car\apps\android
.\gradlew.bat testDebugUnitTest assembleDebug
```

这是手机端单元测试与调试包构建，不会刷写 ESP32 或启动小车。手机端说明见 [Android README](apps/android/README.md)。

## 地图、模型和私密配置

- 只维护 `maps/current/` 这一套实际在用地图；旧实验仍在设备上，原电脑目录在仓库外可恢复归档。
- 模型权重不进入普通 Git。当前机械臂小模型可保留在本地忽略目录；Qwen/YOLO 大模型仍留在 3090。
- `.env.example` 只有变量名；真实密钥、SSH 密码、硬件令牌和服务数据库均禁止提交。
- [源码来源清单](docs/maintenance/source_manifest.json) 记录每个同步文件的远端路径、来源和 SHA-256；整理后新文档不伪装成远端运行文件。

## Git 工作方式

沿用原仓库历史与 `origin`。2026-10-16 交接版本使用独立分支 `handoff/20261016`，不直接覆盖 `main`；重组会表现为大量移动/新增。下一组同学可这样获取交接版：

```bash
git clone --branch handoff/20261016 https://github.com/rales6/bigchuang.git robot-car
cd robot-car
```

从 [GitHub交接说明](docs/handoff_github.md) 开始，完成自己的密钥、模型和设备配置；克隆代码不等于已部署到小车。后续修改先检查差异，再提交到自己的功能分支：

```bash
git status --short
git diff --stat
git add -A
git diff --cached --stat
# 确认没有密钥、日志和错误模型后，再提交、推送到你选择的分支。
```

不要对旧归档目录运行 `git add`，不要把 `runtime/`、录像、模型权重或虚拟环境强行加进 Git。详见 [维护与恢复](docs/maintenance/README.md)。
