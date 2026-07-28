# 3090 推理服务端

## 0. 树莓派摄像头 + Windows 网页三端启动

这个目录是 RTX 3090 Linux 推理服务端。摄像头可以不再放在 Windows 上，而是放到树莓派/小车端：

```text
树莓派：读取摄像头，连接 /camera_ws 上传 JPEG 帧
3090：运行 Qwen、CSRT、YOLO 安全检测、任务状态机，并返回 robot_command
Windows：打开网页，连接 /state 和 /control 显示画面并发送指令，不直接控车
```

3090 启动命令：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
hostname -I
```

树莓派使用 3090 的内网 IP 连接，例如：

```bash
cd ~/linux摄像头
source .venv/bin/activate
python scripts/run_camera_client.py --server ws://<3090内网IP>:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --fps 30 --remote-fps 10 --remote-width 640 --jpeg-quality 70 --overlay-every-n 3
```

接真实小车时，树莓派端加 ESP32 蓝牙输出参数：

```bash
python scripts/run_camera_client.py --server ws://<3090内网IP>:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --remote-fps 10 --robot-mode esp32 --esp32-link ble --esp32-ble-name ESP32-Robot-Car
```

Windows 网页如果走 SSH 隧道，仍然使用本机转发端口：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://127.0.0.1:8001/ws --host 127.0.0.1 --port 7860
```

如果 Windows 能直接访问 3090，可以使用：

```powershell
python .\scripts\run_web_console.py --camera-mode remote --server http://<3090内网IP>:8000 --host 127.0.0.1 --port 7860
```

新增接口说明：

```text
GET  /state       Windows 网页读取最新画面、跟踪状态、机器人任务状态
POST /control     Windows 网页发送新指令、重新框选、急停、手动 ROI
WS   /camera_ws   树莓派摄像头上传 JPEG 帧
WS   /ws          兼容旧版 Windows 本机摄像头直连模式
```

`/camera_ws` 使用低延迟策略：收到树莓派帧后立即缓存最新原始画面并快速返回，后台只处理最新一帧。网页视频刷新不再被 Qwen/CSRT/overlay 的处理速度完全卡住；框选和任务状态仍按 3090 实际推理速度更新。

## 0. 最常用启动方式

这个目录是实际放到 RTX 3090 Linux 机器上运行的服务端代码。3090 负责 Qwen、YOLO、CSRT 跟踪、边界提取、安全仲裁和机器人任务状态机；Windows 端只负责摄像头采集、网页显示、聊天记录和把画面通过 WebSocket 发过来。

在 3090 机器上启动服务：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"ok"}
```

Windows 网页端常用启动命令：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --host 127.0.0.1 --port 7860
```

### 机器人任务流测试

网页对话框里可以直接输入：

```text
捡起垃圾扔进垃圾桶
```

3090 会把这句话解析成 `pick_and_dispose` 任务，并按下面阶段推进：

```text
find_target -> approach_target -> grasp_target -> verify_grasp
-> find_destination -> approach_destination -> release_target -> verify_done -> done
```

当前版本输出的是结构化模拟命令，不会直接驱动真实底盘或机械臂。返回 JSON 中新增两个字段：

```json
{
  "robot_task": {
    "task_type": "pick_and_dispose",
    "phase": "approach_target"
  },
  "robot_command": {
    "mode": "simulated",
    "subsystem": "chassis",
    "action": "approach_target"
  }
}
```

后续接真实机器人时，把 `robot_command` 接到 `chassis_adapter`、`arm_adapter`、`gripper_adapter` 即可。

### 多目标串行处理

现在也支持类似下面的指令：

```text
捡起两个水瓶
```

Qwen 可以返回多个目标框，3090 会按从左到右建立 `target_queue`。执行策略是：

```text
多目标框选 -> 建立队列 -> 激活第 1 个目标 -> CSRT 单目标追踪
-> 模拟抓取完成 -> 标记 done -> 激活第 2 个目标 -> 继续处理
```

也就是说，系统会显示和保存多个候选目标，但机器人控制始终只处理当前 `active` 目标。返回 JSON 中会新增：

```json
{
  "target_queue": [
    {"id": "target_001", "status": "active"},
    {"id": "target_002", "status": "pending"}
  ]
}
```

这样后续接真实机械臂时，不需要同时控制多个物体，只需要按队列逐个抓取。

### YOLO 只做安全障碍物

主目标由 Qwen 框选、CSRT 跟踪；YOLO 不再作为普通物体识别器使用。默认配置只让 YOLO 关注 `stop_labels` 里的安全障碍类别，并且只在危险区域内报告：

```yaml
obstacles:
  frame_interval: 3
  safety_only: true
  report_clear_detections: false
```

如果需要调试 YOLO 的完整检测结果，可以临时改成：

```yaml
obstacles:
  safety_only: false
  report_clear_detections: true
```

正常机器人任务建议保持 `safety_only: true`，避免 YOLO 的类别识别干扰“Qwen 选目标 + CSRT 跟踪”的主链路。

### 视频传输调参

Windows 端可以降低上传帧尺寸、JPEG 质量和 overlay 回传频率：

```powershell
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --remote-fps 8 --remote-width 640 --overlay-every-n 2 --jpeg-quality 70 --overlay-quality 65 --host 127.0.0.1 --port 7860
```

```text
--remote-fps        发给 3090 的最高帧率
--remote-width      发给 3090 前先缩放到的宽度，0 表示不缩放
--overlay-every-n   每 N 个远程帧才回传一次带框 overlay
--jpeg-quality      Windows 上传 JPEG 质量
--overlay-quality   3090 回传 overlay JPEG 质量
```

---

# QwenGroundedTracker Windows v2.0

这是一个面向 Windows 的验证工程，用于测试以下链路：

```text
用户自然语言指令
        ↓
Qwen3-VL 在关键帧中框选具体目标实例
        ↓
CSRT 类别无关跟踪器持续锁定同一个实例
        ↓
GrabCut 近似勾勒目标边界
        ↓
YOLO 独立检测人员和车辆等语义障碍
        ↓
空白二维激光雷达接口 + 安全仲裁器
        ↓
虚拟方向：左转 / 右转 / 前进 / 后退 / 停止
```

本版本不再依赖 YOLO 的固定类别来确定主目标。Qwen 负责根据“右侧的蓝色水杯”“桌上带白色标签的零件”等自然语言选择一个具体实例；后续持续追踪由通用单目标跟踪器完成。YOLO 仅作为语义避障辅助模块。

> 当前运动指令全部是虚拟输出，不会发送给真实底盘。二维激光雷达也是空白占位实现。

## 1. 主要能力

- Qwen3-VL 根据用户指令在单张关键帧中输出目标边界框。
- 多个同类目标存在时，Qwen依据左右、颜色、大小和空间关系选择具体实例。
- CSRT 在后续摄像头帧中持续追踪该实例，不再按“当前最右侧物体”等规则重新选择。
- `logical_target_id` 在 ByteTrack 等检测 ID 之外独立存在，当前项目不依赖 YOLO track ID。
- GrabCut 根据跟踪框近似勾勒目标轮廓；也可切换为仅显示矩形框。
- 颜色直方图、面积和宽高比共同检测明显的跟踪漂移。
- Qwen 推理较慢时，程序保持摄像头刷新并强制虚拟底盘停止。
- Qwen结果属于旧快照时，使用 ORB、模板匹配或归一化坐标尝试在当前帧重新定位。
- YOLO11 分割模型独立检测 `person`、`car` 等语义障碍。
- 二维激光雷达提供统一接口，但当前使用 `NullLidarProvider`，不产生任何扫描数据。
- 目标丢失后可以自动使用“参考目标裁剪图 + 当前画面”重新调用 Qwen。

## 2. 项目结构

```text
QwenGroundedTracker_Windows_v2.0/
├─ configs/
│  ├─ windows_cpu.yaml
│  └─ windows_cuda.yaml
├─ models/
│  ├─ qwen/Qwen3-VL-2B-Instruct/
│  └─ yolo/yolo11n-seg.pt
├─ requirements/
│  ├─ base.txt
│  ├─ qwen.txt
│  ├─ yolo.txt
│  └─ dev.txt
├─ scripts/
│  ├─ setup_venv.ps1
│  ├─ setup_venv.bat
│  ├─ download_models.py
│  ├─ run_grounded_tracking.py
│  ├─ test_qwen_grounding_image.py
│  ├─ test_camera.py
│  └─ smoke_test.py
├─ src/qwen_grounded_tracker/
│  ├─ app.py
│  ├─ domain.py
│  ├─ camera/
│  ├─ grounding/
│  ├─ tracking/
│  ├─ perception/
│  ├─ lidar/
│  ├─ navigation/
│  ├─ safety/
│  └─ ui/
├─ tests/
├─ outputs/
├─ pyproject.toml
└─ README.md
```

核心文件说明：

| 文件 | 作用 |
|---|---|
| `grounding/qwen_grounder.py` | 加载Qwen3-VL并根据指令输出单个目标框 |
| `grounding/parser.py` | 解析Qwen JSON，转换0–1000坐标到摄像头像素 |
| `grounding/worker.py` | 后台执行Qwen推理，主界面继续刷新 |
| `tracking/csrt_tracker.py` | 类别无关的持续单目标追踪 |
| `tracking/identity_guard.py` | 检测明显漂移，避免无声切换到其他物体 |
| `tracking/relocator.py` | 将Qwen旧快照中的框迁移到最新摄像头帧 |
| `tracking/contour_refiner.py` | GrabCut边界近似与矩形框模式 |
| `perception/yolo_obstacles.py` | YOLO语义障碍检测，不负责主目标锁定 |
| `lidar/null_lidar.py` | 空白二维激光雷达占位器 |
| `safety/arbiter.py` | 跟踪、YOLO、雷达和急停的优先级仲裁 |
| `navigation/direction_planner.py` | 根据目标框中心和面积输出虚拟运动方向 |
| `app.py` | 摄像头主循环和所有模块的连接入口 |

## 3. 安装环境

推荐 Python 3.11。

### 3.1 CPU验证环境

在项目根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_venv.ps1 -TorchBackend cpu -WithDev
```

CMD 中可以使用：

```bat
scripts\setup_venv.bat cpu
```

### 3.2 RTX 4090环境

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_venv.ps1 -TorchBackend cu128 -WithDev
```

根据本机 PyTorch 支持情况，也可以使用：

```powershell
.\scripts\setup_venv.ps1 -TorchBackend cu126 -WithDev
```

### 3.3 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

检查环境：

```powershell
python .\scripts\smoke_test.py
```

应重点确认：

```text
CSRT available: True
```

如果为 `False`，说明安装的是普通 `opencv-python`，而不是 `opencv-contrib-python`。

## 4. 下载模型

### 4.1 下载Qwen3-VL

Hugging Face：

```powershell
python .\scripts\download_models.py --qwen --source huggingface
```

ModelScope：

```powershell
python .\scripts\download_models.py --qwen --source modelscope
```

目标目录：

```text
models\qwen\Qwen3-VL-2B-Instruct
```

### 4.2 下载YOLO障碍检测模型

```powershell
python .\scripts\download_models.py --yolo
```

目标文件：

```text
models\yolo\yolo11n-seg.pt
```

YOLO不是主目标跟踪器；关闭YOLO障碍模块后，Qwen + CSRT目标追踪仍可独立运行。

## 5. 修改摄像头ID

CPU配置文件：

```text
configs\windows_cpu.yaml
```

修改：

```yaml
camera:
  source: 0
```

第二个摄像头通常为：

```yaml
camera:
  source: 1
```

先单独测试摄像头：

```powershell
python .\scripts\test_camera.py --camera 0
```

## 6. 单张图片测试Qwen框选

先准备图片，例如：

```text
assets\test_image.jpg
```

CPU测试：

```powershell
python .\scripts\test_qwen_grounding_image.py --image .\assets\test_image.jpg --model .\models\qwen\Qwen3-VL-2B-Instruct --device cpu --command "框选右侧的蓝色水杯"
```

成功后输出图片保存在：

```text
outputs\qwen_grounding_result.jpg
```

## 7. 连续摄像头目标锁定

### 7.1 CPU模式

建议使用单行PowerShell命令，避免反引号后空格导致语法错误：

```powershell
python .\scripts\run_grounded_tracking.py --config .\configs\windows_cpu.yaml --command "持续追踪右侧的蓝色水杯，并画出它的边缘"
```

### 7.2 CUDA模式

```powershell
python .\scripts\run_grounded_tracking.py --config .\configs\windows_cuda.yaml --command "持续追踪右侧的蓝色水杯，并画出它的边缘"
```

### 7.3 多个同类目标

例如左右各有一个水杯：

```powershell
python .\scripts\run_grounded_tracking.py --config .\configs\windows_cpu.yaml --command "只锁定并持续追踪右侧带白色标签的蓝色水杯"
```

Qwen只在首次选择时使用“右侧”条件。目标之后移动到左侧，CSRT仍追踪原实例，不会重新选择当前位于右侧的另一个水杯。

## 8. 运行时快捷键

```text
G       使用当前指令重新调用Qwen框选
I       在终端输入新指令
M       手工框选ROI，用于绕过Qwen快速测试Tracker
R       清除当前目标
C       切换 GrabCut边界 / 矩形框
Space   切换人工急停
S       保存当前原始摄像头画面
Q/Esc   退出
```

## 9. 代码运行逻辑

### 9.1 Qwen首次选择目标

程序读取一张关键帧，并要求Qwen只输出一个紧凑JSON：

```json
{
  "found": true,
  "bbox_2d": [x1, y1, x2, y2],
  "coordinate_system": "relative_1000",
  "target_name": "blue cup",
  "confidence": 0.9
}
```

Qwen只负责“用户指的是哪个具体物体”，不参与逐帧控制。

### 9.2 处理Qwen推理延迟

CPU推理可能较慢。Qwen框对应的是提交请求时的旧快照，因此结果返回后依次尝试：

```text
ORB特征匹配
→ 多尺度模板匹配
→ 同分辨率归一化坐标回退
```

CPU测试时仍建议在首次框选期间保持摄像头和目标基本稳定。4090上推理延迟降低后，这个问题会明显减轻。

### 9.3 持续追踪

初始化后，每帧执行：

```text
CSRT更新目标框
→ IdentityGuard检查颜色、面积和宽高比
→ GrabCut更新或传播目标轮廓
→ DirectionPlanner计算虚拟运动方向
```

CSRT不依赖目标类别，因此可以追踪普通YOLO类别表之外的零件、容器或自定义物体。

### 9.4 目标丢失

```text
短暂丢失
→ 安全层STOP

持续丢失达到配置时间
→ 使用原目标参考裁剪图 + 当前画面重新调用Qwen
→ 成功后重新初始化CSRT
```

配置位置：

```yaml
tracking:
  auto_reground: true
  auto_reground_after_seconds: 3.0
```

CPU阶段如不希望自动等待Qwen，可设置：

```yaml
tracking:
  auto_reground: false
```

然后手动按 `G` 重新框选。

## 10. YOLO障碍模块

YOLO只负责具有语义类别的障碍，例如：

```text
person、car、motorcycle、bus、truck、bicycle、dog
```

当危险类别的框中心进入配置的前方区域时，安全层覆盖目标跟踪指令并输出 `STOP`。

配置：

```yaml
obstacles:
  enabled: true
  stop_labels:
    - person
    - car
  danger_zone:
    x1: 0.28
    y1: 0.42
    x2: 0.72
    y2: 1.00
```

这只是单目视觉启发式规则，不能替代真实距离测量。

## 11. 二维激光雷达占位器

当前文件：

```text
src\qwen_grounded_tracker\lidar\null_lidar.py
```

它始终返回：

```text
ready = false
obstacle = false
min_distance_m = null
status = placeholder: no 2D LiDAR data
```

Windows演示配置中：

```yaml
lidar:
  backend: null
  require_ready: false
```

所以空白雷达不会阻止演示运行。

以后接入真实雷达时，实现相同接口：

```python
class RealLidarProvider:
    def open(self) -> None: ...
    def read(self) -> LidarObservation: ...
    def close(self) -> None: ...
```

实机安全配置应改为：

```yaml
lidar:
  require_ready: true
```

## 12. 安全优先级

```text
人工急停
> 二维激光雷达碰撞风险
> YOLO人员/车辆安全
> 目标是否可靠锁定
> 目标方向控制
```

任何安全条件触发时，最终指令都为：

```text
STOP, linear=0, angular=0
```

## 13. 运行测试

```powershell
python -m pytest
```

当前测试覆盖：

- Qwen JSON及坐标解析
- 方向规划
- LiDAR占位器的安全策略
- 目标身份检查
- 旧快照框的归一化迁移

## 14. 当前限制

1. Qwen CPU框选可能需要较长时间，不适合逐帧调用。
2. CSRT在完全遮挡、快速旋转和大尺度变化时可能漂移。
3. GrabCut只是轻量边界近似，不等同于SAM 2视频分割。
4. 单目目标面积只能粗略表示远近。
5. YOLO危险区域没有实际深度信息。
6. 二维激光雷达尚未实现。
7. 底盘、编码器、IMU、机械臂和π0尚未接入本版本。
8. 当前重点是“自然语言选择具体实例并持续锁定”，不是完整自主导航。

## 15. 推荐后续升级

```text
Windows验证：Qwen + CSRT + GrabCut + YOLO障碍 + NullLiDAR
        ↓
4090阶段：Qwen + SAM 2视频Mask传播 + YOLO障碍 + 真LiDAR
        ↓
实机阶段：局部路径规划 + 编码器/IMU + 底盘安全控制
        ↓
抓取阶段：目标停稳 + 深度/标定 + π0或IK机械臂策略
```
# MJPEG 直连流更新

当前单目 3090 服务新增 `GET /stream.mjpg`。树莓派仍然通过 `WS /camera_ws` 上传 JPEG 帧，Windows 网页在远程摄像头模式下会直接用浏览器连接 `http://<3090内网IP>:8000/stream.mjpg` 显示视频；`/state` 默认只返回跟踪、预测、机器人命令等轻量状态，不再把整张图片塞进 JSON/base64。

启动 3090：
```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

Windows 远程摄像头网页：
```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://192.168.55.33:8000/camera_ws --host 127.0.0.1 --port 7860
```

如果走 SSH 隧道，把 `--server` 改成 `ws://127.0.0.1:8001/camera_ws`，网页会自动使用 `http://127.0.0.1:8001/stream.mjpg`。
# 双摄像头左右拼接模式

3090 服务端现在可以接收树莓派上传的 side-by-side 双摄像头画面。当前阶段不做深度估计，只记录双摄元信息，并根据追踪框在拼接图中的位置判断目标来自：

```text
left   目标中心在左半图，小车左摄像头视野
right  目标中心在右半图，小车右摄像头视野
both   追踪框跨过左右中线
```

启动 3090 不需要换命令：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

树莓派 hello 中带 `stereo.enabled=true` 后，`/state` 会新增 `stereo` 字段，`track/predicted_track/measured_track` 里会新增 `camera_view`。Windows 网页会显示 Camera View，并在拼接图中间画一条左右分界线。
