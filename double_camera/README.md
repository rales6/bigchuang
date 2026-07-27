# Double Camera 双目摄像头初步方案

这个目录是双目摄像头的独立工作区，分成三端：

```text
double_camera/
  3090/          3090 推理端启动包装和双目配置
  raspberry_pi/  树莓派/Linux 双摄采集与上传代码
  windows_web/   Windows 网页端启动包装
```

## 当前阶段目标

第一阶段先不做完整双目深度，而是做：

```text
左摄像头 + 右摄像头
-> 树莓派按左右顺序拼接成 side-by-side 宽图
-> 上传给 3090
-> 3090 按一张宽图继续 Qwen / YOLO / CSRT / 任务控制
-> Windows 网页显示宽图和追踪结果
```

这样改动小，能先验证双摄视野、目标不易丢失、左右方向判断更稳定。

## 摄像头编号和小车左右

摄像头编号必须按小车物理方向记录：

```text
--left-camera   小车左侧摄像头，对应拼接图左半边
--right-camera  小车右侧摄像头，对应拼接图右半边
```

如果画面左右反了，不要在 3090 里修，优先交换树莓派启动命令里的两个编号：

```bash
--left-camera 2 --right-camera 0
```

## 为深度估计预留的信息

现在至少记录：

```text
baseline_mm = 两个摄像头光心之间的水平距离，单位 mm
left_camera = 左摄像头 Linux 编号
right_camera = 右摄像头 Linux 编号
layout = side_by_side
```

后续真正做深度，还需要：

```text
左右相机内参 fx/fy/cx/cy
左右相机畸变参数
左右相机外参 R/T
stereoRectify 矫正映射
```

仅知道 baseline 可以先做粗略远近趋势，不能直接做精确抓取距离。

## 推荐启动顺序

### 1. 3090

```bash
cd ~/double_camera/3090
source ~/3090/.venv/bin/activate
python scripts/run_double_camera_server.py --host 0.0.0.0 --port 8000
```

### 2. 树莓派/Linux 双摄上传

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

### 3. Windows 网页

```powershell
cd F:\bigchuang\double_camera\windows_web
python .\scripts\run_double_camera_web.py --server ws://192.168.55.33:8000/camera_ws --host 127.0.0.1 --port 7860 --remote-fps 30
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 初步深度路线

1. 当前：side-by-side 拼接图，先跑通双摄视野。
2. 下一步：保存左右同步帧，用棋盘格做双目标定。
3. 再下一步：3090 或树莓派端根据标定参数拆分左右图，计算同一目标 bbox 的视差。
4. 最后：把深度从相机坐标转换到底盘/机械臂坐标。

