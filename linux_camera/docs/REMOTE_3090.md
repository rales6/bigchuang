# 3090 Linux 远程处理部署方案

## 交互方式

远程模式把原来的单机循环拆成两端：

```text
Windows/Linux 摄像头端
  OpenCV 采集摄像头
  JPEG 压缩当前帧
  WebSocket 发送到 3090
        |
        v
3090 Linux 推理端
  Qwen3-VL 首帧或重定位时做目标 grounding
  CSRT 持续跟踪同一实例
  GrabCut/矩形框生成目标边界
  YOLO11 检测人、车等语义障碍
  SafetyArbiter 输出最终 STOP/FORWARD/TURN 指令
        |
        v
摄像头端
  接收 JSON 指令和可选标注 JPEG
  本地显示标注画面
```

摄像头端只需要能打开本机摄像头，不需要安装 Qwen 或 YOLO。3090 端不直接碰摄像头驱动，只处理网络传来的图像。

## 网络协议

使用 WebSocket，默认服务地址：

```text
ws://<3090机器IP>:8000/ws
```

客户端首先发送：

```json
{"type":"start","instruction":"持续跟踪右侧的蓝色水杯","return_overlay":true}
```

之后每一帧发送 JPEG 二进制。服务端每帧返回 JSON：

```json
{
  "type": "result",
  "track": {"visible": true, "logical_target_id": "target_001"},
  "guidance": {"direction": "FORWARD", "linear": 0.08, "angular": 0.0},
  "safety": {"blocked": false, "reasons": []},
  "overlay_jpeg_b64": "..."
}
```

控制消息仍走同一条 WebSocket：

```text
G       重新调用 Qwen grounding
I       输入新自然语言指令
M       在摄像头端手动画 ROI
R       清除当前目标
C       切换 GrabCut 边界/矩形框
Space   切换急停
Q/Esc   退出
```

## 3090 Linux 服务端

建议 Python 3.11。先安装 CUDA 版 PyTorch，再安装本项目依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/remote.txt
```

确认模型放在：

```text
models/qwen/Qwen3-VL-2B-Instruct
models/yolo/yolo11n-seg.pt
```

启动服务：

```bash
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

如果防火墙开启，需要允许摄像头机器访问 TCP 8000。

## Windows 摄像头客户端

客户端不需要 Qwen/YOLO，但为了简单可以直接安装远程依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements\remote.txt
```

运行：

```powershell
python .\scripts\run_camera_client.py --server ws://<3090机器IP>:8000/ws --camera 0 --backend dshow --width 640 --height 480 --command "持续跟踪右侧的蓝色水杯"
```

## Linux 摄像头客户端

```bash
python scripts/run_camera_client.py --server ws://<3090机器IP>:8000/ws --camera 0 --backend v4l2 --width 640 --height 480 --command "持续跟踪右侧的蓝色水杯"
```

也可以把 `--camera` 换成 RTSP/HTTP 视频流 URL。

## 实时性建议

- 局域网内优先用有线网络。
- 起步用 `640x480`、`--jpeg-quality 70~80`。
- Qwen 只在初始选择、手动重新选择或丢失后重定位时较慢；跟踪阶段主要是 CSRT + YOLO。
- 如果只需要运动 JSON，不需要回传标注图，可以给客户端加 `--no-overlay` 降低带宽。
- 目前服务端按单摄像头连接设计；多摄像头建议每路开一个服务端端口或后续扩展 session 管理。
