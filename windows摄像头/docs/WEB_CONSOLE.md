# Windows 摄像头网页上位机

## 快速启动

3090 端先启动：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/run_3090_server.py --config configs/linux_3090_remote.yaml --host 0.0.0.0 --port 8000
```

Windows 端启动：

```powershell
cd F:\bigchuang\windows摄像头
.\.venv\Scripts\Activate.ps1
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

网页上位机会自动检查 `http://127.0.0.1:8001/health`。如果本地 8001 隧道未开启，会自动运行：

```powershell
.\scripts\start_3090_tunnel.ps1 -LocalPort 8001
```

## 对话记录

网页支持：

- 新建对话
- 继续历史对话
- 重命名历史对话
- 删除历史对话

聊天记录保存到：

```text
outputs/chat_history.json
```

## 工作方式

- Windows 端只负责摄像头采集、网页显示、聊天记录和 WebSocket 转发。
- 3090 端负责 Qwen、YOLO、CSRT、GrabCut、安全仲裁。
- 第一条任务指令发送后，Windows 才会连接 3090 并持续发送摄像头帧。
- 3090 返回带框选/轮廓/状态叠加的 JPEG，网页实时显示。

## 刷新率

页面顶部显示：

```text
本地摄像头 FPS / 3090 返回 FPS
```

降低远端负载：

```powershell
python .\scripts\run_web_console.py --server ws://127.0.0.1:8001/ws --camera 1 --backend dshow --width 640 --height 480 --remote-fps 5 --jpeg-quality 65 --overlay-quality 65 --host 127.0.0.1 --port 7860
```
