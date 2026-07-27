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
