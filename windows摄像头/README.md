# Windows 摄像头网页上位机

这个目录只放 Windows 网页上位机代码。默认模式仍然支持 Windows 本机摄像头；如果摄像头放到树莓派上，Windows 端只负责显示网页、保存聊天记录、发送 Qwen/机器人任务指令，不再打开本机摄像头。

## 0. 树莓派摄像头三端启动方式

三端分工如下：

```text
树莓派摄像头 -> WebSocket 上传帧 -> 3090 推理服务
Windows 网页 -> HTTP 状态/控制 -> 3090 推理服务
3090 robot_command -> WebSocket 返回 -> 树莓派执行底盘/机械臂命令
```

先在 3090 上启动服务：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

再在树莓派上启动摄像头上传：

```bash
cd ~/linux摄像头
source .venv/bin/activate
python scripts/run_camera_client.py --server ws://192.168.55.33:8000/camera_ws --camera 0 --backend v4l2 --width 640 --height 480 --fps 30 --remote-fps 10 --remote-width 640 --jpeg-quality 70 --overlay-every-n 3
```

最后在 Windows 上启动网页控制台。若 Windows 仍通过 SSH 隧道访问 3090，继续使用 `127.0.0.1:8001`：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://127.0.0.1:8001/ws --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果 Windows 可以直接访问 3090，也可以把 `--server` 改成：

```powershell
--server http://<3090内网IP>:8000
```

树莓派模式下，网页上的“重新调用 Qwen、切换边界、急停、应用手动框选”等按钮会通过 `/control` 发给 3090；画面和状态通过 `/state` 从 3090 读取。
真实小车控制不从 Windows 直接发出，而是由 3090 在处理每帧后把 `robot_command` 返回给树莓派，树莓派再通过 BLE/UART 调用 ESP32 控制底盘和机械臂。

## 1. 最常用启动方式

先确认 3090 端已经启动服务：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

然后在 Windows 端启动网页上位机：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果 `ws://127.0.0.1:8001/ws` 不通，网页上位机会自动运行：

```powershell
.\scripts\start_3090_tunnel.ps1 -LocalPort 8001
```

也就是说，通常不需要单独开隧道窗口。若想手动管理隧道，在启动网页时加：

```powershell
--no-auto-tunnel
```

## 2. 首次安装

```powershell
cd F:\bigchuang\windows摄像头
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\client.txt
```

如果摄像头不是 `1`，把启动命令里的 `--camera 1` 改成 `0` 或其他编号。

## 3. 网页功能

- 左侧工作区：摄像头画面、手动 ROI、控制按钮、运行状态。
- 右侧对话区：类似 GPT 的对话界面。
- 支持新对话、继续历史对话、重命名历史对话、删除历史对话。
- 输入 `捡起垃圾扔进垃圾桶` 会触发 3090 端的 `pick_and_dispose` 任务流。
- 页面指标卡会显示 `Robot Task` 和 `Robot Command`，目前是模拟命令，还没有直接驱动真实机器人。
- 输入 `捡起两个水瓶` 时，3090 会尝试建立多目标队列，页面的 `Target Queue` 会显示当前 active、done 和总数。
- 聊天记录保存在本地 JSON：

```text
outputs/chat_history.json
```

历史对话操作：

- `New chat`：新建一条空对话。
- `Continue`：继续选中的历史对话，后续指令会追加到同一个聊天记录里。
- `Rename`：重命名历史对话标题。
- `Delete`：删除历史对话，同时从本地 JSON 里移除。

## 4. 常用调参

降低发给 3090 的帧率和 JPEG 质量：

```powershell
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --remote-fps 5 --remote-width 640 --overlay-every-n 2 --jpeg-quality 65 --overlay-quality 60 --host 127.0.0.1 --port 7860
```

页面顶部 FPS 显示含义：

```text
Windows 本地摄像头 FPS / 3090 返回 FPS
```

如果本地 FPS 高、3090 返回 FPS 低，瓶颈通常在 3090 推理、YOLO/跟踪、SSH 隧道或叠加图编码；如果本地 FPS 也低，优先检查 Windows 摄像头和 OpenCV 后端。

传输参数说明：

```text
--remote-width      上传到 3090 前缩放视频宽度；0 表示不缩放。
--overlay-every-n   每 N 帧才让 3090 回传一次带框画面。
--jpeg-quality      上传 JPEG 质量，越低越省带宽。
--overlay-quality   3090 回传 overlay JPEG 质量，越低越省带宽。
```

## 5. 手动隧道备用命令

```powershell
cd F:\bigchuang\windows摄像头
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_3090_tunnel.ps1 -LocalPort 8001
```

隧道健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8001/health
```

返回 `{"status":"ok"}` 表示 Windows 已经能通过隧道访问 3090 服务。
# MJPEG 直连流更新

远程摄像头模式下，网页视频现在默认直接连接 3090 的 `http://<3090内网IP>:8000/stream.mjpg`。Windows 本地服务只负责页面、聊天记录、按钮控制和 `/state` 状态轮询；视频不再从 `/state` JSON/base64 中取图，因此延迟和 CPU 开销会更低。

常用启动：
```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://192.168.55.33:8000/camera_ws --host 127.0.0.1 --port 7860
```

如果通过 SSH 隧道访问 3090：
```powershell
python .\scripts\run_web_console.py --camera-mode remote --server ws://127.0.0.1:8001/camera_ws --host 127.0.0.1 --port 7860
```

如需手动指定视频流地址，可额外加 `--video-url http://<地址>:<端口>/stream.mjpg`。
# 双摄像头网页显示

双摄像头模式下，Windows 端仍然只启动网页控制台，不直接打开摄像头。树莓派会把左右摄像头拼成一张 side-by-side 图上传给 3090，网页通过 `stream.mjpg` 显示拼接画面，并通过 `/state` 显示目标位于 `left/right/both`。

启动网页：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --camera-mode remote --server ws://192.168.55.33:8000/camera_ws --host 127.0.0.1 --port 7860
```

页面指标里的 `Camera View` 表示当前追踪目标位于左摄像头、右摄像头，还是跨过两侧中线。画面中间的虚线是左右摄像头分界线，不表示深度。黄/绿实线框是参与控制的主追踪框；蓝色虚线框是 Qwen 在另一侧摄像头真实识别到目标时返回的 paired track，只用于观察参考，不直接参与小车控制。只有一个摄像头看到目标时，网页只显示那一侧的锁定框。
