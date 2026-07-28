# 树莓派摄像头上传端

这个目录放在树莓派或 Linux 摄像头端运行。它只负责读取摄像头、压缩成 JPEG、通过 WebSocket 上传给 3090；Qwen、YOLO、CSRT 跟踪、机器人任务状态都不在树莓派上运行。

## 0. 最常用启动方式

先确认 3090 服务已经启动：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

在 3090 上查看内网 IP：

```bash
hostname -I
```

然后在树莓派上启动摄像头上传：

```bash
cd ~/linux摄像头
source .venv/bin/activate
python scripts/run_camera_client.py --server ws://<3090内网IP>:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --fps 30 --remote-fps 10 --remote-width 640 --jpeg-quality 70 --overlay-every-n 3
```

默认不会弹出 OpenCV 窗口，适合放到小车上长期运行。需要在树莓派本地调试画面时加：

```bash
--window
```

默认 `--robot-mode dry-run` 只打印 3090 返回的小车控制命令，不会驱动真实电机。确认画面、框选和任务阶段稳定后，再切换到 ESP32 蓝牙控制：

```bash
python scripts/run_camera_client.py --server ws://<3090内网IP>:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --remote-fps 10 --robot-mode esp32 --esp32-link ble --esp32-ble-name ESP32-Robot-Car
```

如果 BLE 连接不稳定，也可以使用 `auto`，让树莓派优先尝试 UART，失败后再切 BLE：

```bash
python scripts/run_camera_client.py --server ws://<3090内网IP>:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --remote-fps 10 --robot-mode esp32 --esp32-link auto --esp32-uart-port /dev/serial0 --esp32-ble-name ESP32-Robot-Car
```

## 1. 首次安装

```bash
cd ~/linux摄像头
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/client.txt
```

如果使用树莓派 CSI 摄像头且 OpenCV 无法直接打开 `--camera 0`，可以先确认系统摄像头节点：

```bash
ls /dev/video*
```

## 2. 常用参数

```text
--server           3090 的摄像头 WebSocket 地址，通常是 ws://<3090内网IP>:8000/camera_ws
--camera           OpenCV 摄像头编号或视频流 URL
--backend          Linux/树莓派一般使用 v4l2
--remote-fps       上传给 3090 的最高帧率
--remote-width     上传前缩放到的宽度；0 表示不缩放
--remote-height    上传前缩放到的高度；0 表示按宽度保持比例
--jpeg-quality     树莓派上传 JPEG 质量，越低越省带宽
--overlay-every-n  3090 每 N 帧回传一次带框画面
--no-overlay       树莓派不请求 overlay，Windows 仍可通过 /state 看 3090 最近状态
--window           在树莓派本地显示调试窗口
--robot-mode       小车命令输出方式：none、dry-run、serial、esp32
--robot-debug-log  dry-run 模式下打印小车控制命令；默认不打印
--esp32-link       ESP32 链路：ble、uart、auto
--esp32-ble-name   BLE 广播名，默认 ESP32-Robot-Car
--esp32-ble-address 直接指定 BLE 地址，可选
--esp32-uart-port  UART 端口，默认 /dev/serial0
--esp32-uart-baudrate UART 波特率，默认 230400
--esp32-ttl-ms     set_twist 速度命令 TTL，默认 600 ms
--gripper-joint    夹爪关节编号，默认 5
--gripper-open-us  夹爪打开脉宽，默认 1200 us
--gripper-close-us 夹爪闭合脉宽，默认 1550 us
--robot-command-rate 发送给下位机的最高命令频率
--robot-max-linear  底盘最大线速度限幅
--robot-max-angular 底盘最大角速度限幅
```

建议先用 `640x480`、`remote-fps 10`、`jpeg-quality 70` 测通，再根据网络和画面流畅度提高参数。

## 3. 小车控制命令流

控制闭环推荐放在树莓派侧：

```text
3090 返回 robot_command/guidance/safety
-> 树莓派 RobotCommandDispatcher 做安全限幅和急停判断
-> raspberry_pi.esp32.Esp32Client 通过 BLE/UART 发给 ESP32
-> ESP32 驱动底盘、舵机、夹爪
```

`dry-run` 模式会打印树莓派准备发给 ESP32 的结构化命令，示例：

```json
{
  "type": "robot_control",
  "chassis": {
    "enabled": true,
    "linear": 0.08,
    "angular": -0.22,
    "linear_mm_s": 80,
    "angular_mrad_s": -220,
    "ttl_ms": 600
  },
  "arm": {"action": "hold"},
  "task": {"type": "pick_and_dispose", "phase": "approach_target", "completed": false},
  "source": {"subsystem": "chassis", "action": "approach_target", "direction": "FORWARD"}
}
```

当 3090 返回 `safety.blocked=true`、网页急停打开、Qwen 正在框选、目标丢失或任务进入视觉等待阶段时，树莓派会发送停车命令：

```json
{"type":"robot_control","chassis":{"enabled":true,"linear":0.0,"angular":0.0,"linear_mm_s":0,"angular_mrad_s":0,"ttl_ms":600},"arm":{"action":"hold"}}
```

`esp32` 模式不会发送 JSON 行，而是按你的协议直接调用 ESP32 客户端 API：

```python
client.set_twist(linear_mm_s, angular_mrad_s, ttl_ms=600)
client.stop()
client.set_arm_joints([(5, 1550)], duration_ms=800)
client.cancel_all()
```

## 4. Windows 网页控制

Windows 不再读取摄像头，启动远程摄像头模式：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://127.0.0.1:8001/ws --host 127.0.0.1 --port 7860
```

网页打开后，聊天框里输入“捡起垃圾扔进垃圾桶”等指令，Windows 会把指令发给 3090；树莓派只持续上传画面。
## 5. Low-latency camera upload

If the web page is several seconds behind the real scene, the most common cause is stale camera frames queued by V4L2 while the 3090 is processing. Use a tiny camera buffer and discard queued frames before each upload:

```bash
python scripts/run_camera_client.py --server ws://192.168.55.33:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --fps 30 --camera-buffer 1 --drop-stale-frames 5 --remote-fps 10 --remote-width 416 --jpeg-quality 35 --overlay-every-n 1 --overlay-quality 35
```

The client prints one line per second like:

```text
[Remote] track=tracking guidance=FORWARD blocked=False rtt=180ms jpeg=18KB
```

`rtt` is the WebSocket round trip from Raspberry Pi to the 3090 and back. If `rtt` is low but the browser is still delayed, check the Windows web console. If `rtt` is close to the visible delay, lower `remote-width`, lower JPEG quality, or reduce Qwen/overlay work on the 3090.
# MJPEG 直传采集脚本

如果 `run_camera_client.py` 已经显示 `effective ... fourcc=MJPG`，但 `read` 仍然在 60ms 左右，可以改用直传脚本。它通过 `ffmpeg` 从 V4L2 摄像头读取摄像头硬件输出的 MJPEG 压缩帧，直接发给 3090，跳过 OpenCV 解码和二次 JPEG 编码。

树莓派先确认有 ffmpeg：
```bash
ffmpeg -version
```

启动：
```bash
cd ~/dachuang/linux_camera
source .venv/bin/activate
python scripts/run_mjpeg_camera_client.py --server ws://192.168.55.33:8000/camera_ws --camera 0 --width 640 --height 480 --fps 30 --remote-fps 30 --no-overlay --robot-mode none
```

这个脚本不支持 `--remote-width 320` 这种 OpenCV 缩放，因为它不解码图像；需要摄像头实际支持的 MJPG 分辨率，例如 `640x480`、`800x600`、`1280x720`。如果 MJPEG 直传能达到 25~30FPS，再把 `--robot-mode none` 换成 `dry-run` 或 `esp32`。
# 双摄像头左右拼接模式

当前小车使用同一高度、水平左右放置的双摄像头时，不单独做深度估计。树莓派端把左摄像头和右摄像头并行读取，然后拼成一张 side-by-side 图上传给 3090：左半边固定代表小车左摄像头，右半边固定代表小车右摄像头。

推荐先让每个摄像头独立保持约 15FPS，上传拼接图也使用 15FPS：

```bash
cd ~/dachuang/linux_camera
source .venv/bin/activate
python scripts/run_camera_client.py \
  --server ws://192.168.55.33:8000/camera_ws \
  --left-camera 0 \
  --right-camera 2 \
  --swap-cameras \
  --baseline-mm 120 \
  --backend v4l2 \
  --width 640 --height 480 --fps 30 \
  --fourcc MJPG \
  --remote-fps 15 \
  --remote-width 960 \
  --jpeg-quality 45 \
  --no-overlay \
  --robot-mode none
```

日志里会显示 `left_fps` 和 `right_fps`。两者都接近 15 时，说明左右摄像头采集达标。如果画面左右对应反了，加 `--swap-cameras`；如果已经正确，就去掉这个参数。确认画面、追踪和小车指令稳定后，再把 `--robot-mode none` 换成 `dry-run` 或 `esp32`。
