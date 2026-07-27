# Windows 双目网页端

Windows 端继续复用现有网页控制台，只是默认工作在 remote camera 模式。

## 启动

```powershell
cd F:\bigchuang\double_camera\windows_web
python .\scripts\run_double_camera_web.py --server ws://192.168.55.33:8000/camera_ws --host 127.0.0.1 --port 7860 --remote-fps 30
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 显示说明

网页显示的是树莓派上传给 3090 的 side-by-side 宽图：

```text
左半边 = 小车左侧摄像头
右半边 = 小车右侧摄像头
```

如果画面相反，修改树莓派启动命令中的 `--left-camera` 和 `--right-camera`，不要在网页里临时修。

