# 3090 双目端

当前 3090 双目端复用原来的 `~/3090` 推理服务。树莓派会把左右摄像头拼成一张宽图，因此 3090 仍然只接收一个 `/camera_ws` 视频流。

## 启动

```bash
cd ~/double_camera/3090
source ~/3090/.venv/bin/activate
python scripts/run_double_camera_server.py --host 0.0.0.0 --port 8000
```

默认使用：

```text
configs/double_camera_3090_remote.yaml
```

## 当前处理方式

```text
side-by-side 宽图
-> Qwen 目标框选
-> CSRT 追踪
-> YOLO 安全障碍
-> robot_command 返回树莓派
```

## 深度预留

树莓派 hello 消息会包含：

```json
{
  "stereo": {
    "layout": "side_by_side",
    "left_camera": "0",
    "right_camera": "2",
    "baseline_mm": 120.0
  }
}
```

后续可以在 3090 的 `/camera_ws` 收到 hello 时保存这些参数，再结合标定文件做深度估计。

