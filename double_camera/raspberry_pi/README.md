# 树莓派/Linux 双摄上传端

这个端负责读取两个 Linux 摄像头，并按“小车左侧在左半图、小车右侧在右半图”的规则拼接上传。

## 摄像头编号

先查看摄像头：

```bash
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video2 --list-formats-ext
```

编号规则：

```text
--left-camera   小车左侧摄像头
--right-camera  小车右侧摄像头
```

如果网页中左侧显示的是右摄像头，直接交换两个参数。

## 启动

```bash
cd ~/double_camera/raspberry_pi
source ~/dachuang/linux_camera/.venv/bin/activate
python scripts/run_double_camera_client.py \
  --server ws://192.168.55.33:8000/camera_ws \
  --left-camera 0 \
  --right-camera 2 \
  --baseline-mm 120 \
  --width 640 --height 480 --fps 30 \
  --fourcc MJPG \
  --remote-fps 20 \
  --remote-width 960 \
  --jpeg-quality 45 \
  --no-overlay
```

## 日志含义

```text
sent_fps    上传到 3090 的拼接图 FPS
recv_fps    收到 3090 返回状态的 FPS
read_l/r    左/右摄像头读取耗时
encode      JPEG 编码耗时
send        WebSocket 发送耗时
avg_jpeg    每帧 JPEG 大小
```

如果 `read_l/read_r` 很高，说明瓶颈在摄像头读取，需要尝试 `--fourcc MJPG`、降低采集分辨率或检查曝光。

## 深度准备

当前脚本会把以下信息发给 3090：

```text
left_camera
right_camera
baseline_mm
layout=side_by_side
```

后续标定后，可以增加：

```text
left_intrinsic
right_intrinsic
distortion
R/T
rectification maps
```

