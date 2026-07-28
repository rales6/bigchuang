# 双目摄像头半自动标定

这套脚本用于给小车左右两个摄像头做 OpenCV 双目标定，输出后续粗略测距需要的内参、外参、基线和矫正矩阵。

## 1. 需要准备的物理参数

必须知道：

- 棋盘格内角点数量，例如 `9x6`。注意不是格子数量，而是黑白交界的内角点数量。
- 棋盘格每个方格边长，例如 `25mm`。
- 左右摄像头编号，例如 `/dev/video0` 是小车左摄像头，`/dev/video2` 是小车右摄像头。

建议记录：

- 左右摄像头光心大致水平距离，也就是物理基线，用来和标定输出的 `baseline_mm_norm` 互相校验。
- 摄像头安装高度、俯仰角。后续如果做地面目标测距，可以作为辅助约束。

## 2. 在树莓派采集标定图片

在树莓派或 Linux 摄像头端运行：

```bash
cd ~/dachuang/linux_camera
source .venv/bin/activate
python scripts/capture_stereo_calibration.py \
  --left-camera 0 \
  --right-camera 2 \
  --backend v4l2 \
  --width 640 --height 480 --fps 30 \
  --fourcc MJPG \
  --board-cols 9 \
  --board-rows 6 \
  --output outputs/stereo_calibration_samples \
  --count 35
```

如果左右画面反了，加：

```bash
--swap-cameras
```

如果没有桌面窗口，可以加：

```bash
--headless
```

采集建议：

- 至少 25 对，推荐 35 到 50 对。
- 棋盘格要覆盖画面左上、右上、左下、右下、中心、近处、远处。
- 棋盘格要有不同倾斜角度，不要全部正对摄像头。
- 每张图必须左右摄像头都能完整看到棋盘格。
- 采集时不要移动太快，避免运动模糊。

脚本会保存：

```text
outputs/stereo_calibration_samples/
  left/left_001.jpg
  right/right_001.jpg
```

## 3. 把图片传到 3090

可以直接复制整个采集目录到 3090：

```bash
scp -r outputs/stereo_calibration_samples sjtu@192.168.55.33:~/3090/outputs/
```

如果你在 Windows 上中转，也可以把这个目录放到：

```text
F:\bigchuang\3090\outputs\stereo_calibration_samples
```

## 4. 在 3090 上计算标定

在 3090 端运行：

```bash
cd ~/3090
source .venv/bin/activate
python scripts/calibrate_stereo_from_images.py \
  --input outputs/stereo_calibration_samples \
  --board-cols 9 \
  --board-rows 6 \
  --square-mm 25 \
  --output configs/stereo_calibration.yaml \
  --json-output configs/stereo_calibration.json \
  --preview outputs/stereo_rectified_preview.jpg
```

输出文件：

- `configs/stereo_calibration.yaml`：主配置，后续测距读取这个。
- `configs/stereo_calibration.json`：方便其他程序读取。
- `outputs/stereo_rectified_preview.jpg`：左右图极线校正预览。

重点看终端输出：

```text
valid_pairs=35
stereo_rms=...
baseline_norm=... mm
```

判断标准：

- `valid_pairs` 建议大于 25。
- `baseline_norm` 应该接近你实际量到的左右摄像头距离。
- `stereo_rms` 越小越好。粗略小车测距通常先争取小于 1.0 到 1.5。
- 如果 `baseline_norm` 明显离谱，比如实际 120mm 但输出 30mm 或 400mm，通常是 `square-mm` 填错、棋盘格角点数量填错、左右图不同步或左右摄像头反了。

## 5. 检查标定效果

用任意一对左右图做矫正检查：

```bash
python scripts/check_stereo_calibration.py \
  --calibration configs/stereo_calibration.yaml \
  --left outputs/stereo_calibration_samples/left/left_001.jpg \
  --right outputs/stereo_calibration_samples/right/right_001.jpg \
  --output outputs/stereo_check_rectified.jpg
```

打开 `outputs/stereo_check_rectified.jpg`，同一个棋盘格角点在左右图中应该基本落在同一条水平线上。

如果明显上下错开：

- 两个摄像头固定不平行或有明显旋转。
- 棋盘格样本太少或角度变化不够。
- 左右摄像头顺序反了。
- 左右图不是同一时刻采集，棋盘格移动造成误差。

## 6. 后续接入测距

粗略距离公式：

```text
Z_mm = fx_px * baseline_mm / disparity_px
```

但正式使用时建议先做极线校正，然后用输出里的 `Q` 矩阵或校正后的 `P1/P2` 计算。目标框层面的粗测距可以先用：

- 左右目标框中心点的 x 差作为 `disparity_px`。
- 框面积作为辅助，避免视差很小时距离爆炸。
- 只在左右都可靠看到目标时输出距离；只有一侧看到时只做方向控制，不做距离闭环。

当前建议先把 `stereo_calibration.yaml` 产出并检查通过，再把它接进 3090 的双目融合控制。
