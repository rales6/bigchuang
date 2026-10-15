# 半自动双目标定工具

这个文件夹是独立的双目摄像头标定工具，不依赖主项目的运行流程。

## 设备分工

| 设备 | 运行脚本 | 作用 |
| --- | --- | --- |
| 树莓派 / Linux 摄像头端 | `raspberry_pi/capture_stereo_calibration.py` | 同时读取左右摄像头，自动采集棋盘格图片 |
| 3090 主机 | `3090/calibrate_stereo_from_images.py` | 根据左右图片计算双目标定参数 |
| 3090 主机 | `3090/check_stereo_calibration.py` | 生成极线校正检查图，确认标定是否可用 |
| Windows | 不必须运行脚本 | 只用于中转文件或查看输出图片 |

## 1. 准备棋盘格参数

你需要知道两个参数：

- `board-cols`：棋盘格每行内角点数量，不是格子数。
- `board-rows`：棋盘格每列内角点数量。
- `square-mm`：每个方格的实际边长，单位 mm。

例如常见棋盘格是 `9 x 6` 内角点，方格边长 `25mm`。

## 2. 树莓派采集图片

在树莓派上运行：

```bash
cd ~/stereo_calibration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python raspberry_pi/capture_stereo_calibration.py \
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

如果左右摄像头画面反了，加：

```bash
--swap-cameras
```

如果树莓派没有桌面窗口，加：

```bash
--headless
```

采集建议：

- 至少采集 25 对，推荐 35 到 50 对。
- 棋盘格要出现在画面中心、四角、近处、远处。
- 棋盘格要有不同倾斜角度，不要每张都正对摄像头。
- 左右摄像头必须同时完整看到棋盘格。
- 采集时不要移动太快，避免模糊。

采集结果：

```text
outputs/stereo_calibration_samples/
  left/left_001.jpg
  right/right_001.jpg
```

## 3. 把图片传到 3090

在树莓派上执行：

```bash
scp -r outputs/stereo_calibration_samples sjtu@192.168.55.33:~/stereo_calibration/outputs/
```

如果 3090 的 IP 不是 `192.168.55.33`，先在 3090 上运行：

```bash
hostname -I
```

## 4. 3090 计算标定

在 3090 上运行：

```bash
cd ~/stereo_calibration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python 3090/calibrate_stereo_from_images.py \
  --input outputs/stereo_calibration_samples \
  --board-cols 9 \
  --board-rows 6 \
  --square-mm 25 \
  --output outputs/stereo_calibration.yaml \
  --json-output outputs/stereo_calibration.json \
  --preview outputs/stereo_rectified_preview.jpg
```

输出文件：

- `outputs/stereo_calibration.yaml`：后续 3090 测距主要读取这个。
- `outputs/stereo_calibration.json`：给其他程序读取更方便。
- `outputs/stereo_rectified_preview.jpg`：极线校正预览图。

重点看终端输出：

```text
valid_pairs=35
stereo_rms=...
baseline_norm=... mm
```

判断标准：

- `valid_pairs` 建议大于 25。
- `baseline_norm` 应该接近你实际量到的左右摄像头水平距离。
- `stereo_rms` 越小越好，粗略测距建议先做到小于 `1.0 ~ 1.5`。

## 5. 3090 检查标定

在 3090 上运行：

```bash
python 3090/check_stereo_calibration.py \
  --calibration outputs/stereo_calibration.yaml \
  --left outputs/stereo_calibration_samples/left/left_001.jpg \
  --right outputs/stereo_calibration_samples/right/right_001.jpg \
  --output outputs/stereo_check_rectified.jpg
```

打开 `outputs/stereo_check_rectified.jpg` 看效果：

- 同一个棋盘格角点应该基本在同一条水平线上。
- 如果左右明显上下错开，说明标定质量不够，需要重新采集。

## 6. 常见问题

### baseline 明显不对

检查：

- `--square-mm` 是否填成真实方格边长。
- `--board-cols` 和 `--board-rows` 是否填的是内角点数量。
- 左右摄像头是否反了。
- 左右图片是不是同一时刻采集的。

### stereo_rms 很大

重新采集，增加不同角度和不同位置的棋盘格图片。不要只在画面中心拍，也不要全是同一个距离。

### 后续如何测距

粗略公式：

```text
Z_mm = fx_px * baseline_mm / disparity_px
```

更稳的方式是先用标定输出里的 `R1/R2/P1/P2/Q` 做极线校正，再用左右目标框中心点的 x 差计算距离。
