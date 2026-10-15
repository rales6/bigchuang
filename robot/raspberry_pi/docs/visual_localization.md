# 视觉初始粗定位（Visual Place Recognition）

视觉模块只缩小已有雷达主动定位的搜索范围，不直接把一张照片当作绝对位姿。匹配失败、
参考库缺失、图片损坏、特征过少或 OpenCV 不可用时，任务应记录 warning 并继续原有的
全地图主动定位。

## 流程

原流程：相机自检 → 全地图雷达候选 → 主动移动消歧 → 稳定位姿 → 导航。

加入视觉后：相机自检 → 当前双目/单目照片 → ORB + ratio test + RANSAC → Top-3
地点候选 → 必要时安全移动 10–30 cm 后再次拍摄 → 0.6 当前帧 + 0.4 历史帧融合 →
仅在 Top-3 附近运行雷达几何定位。视觉不可靠时恢复全地图雷达搜索。

`score` 是由 RANSAC 内点数、内点比例、通过 ratio test 的匹配数和灰度直方图一致性
共同得到的归一化质量分数，不是概率。只有 Top-1 同时通过最低分、最低内点数和相对
Top-2 的 margin 才会标为 `reliable=true`。

## 参考照片采集

- 每个候选位置建议 4 个方向（前、左、右、后），至少前/左/右各一张；双相机则两路
  同时保存。
- 普通房间候选间距建议 0.6–1.0 m；重复走廊或相似柜体附近缩短到 0.4–0.6 m。
- 建议 1280×720 或 1920×1080；建库时最长边会缩放到 960 像素，避免树莓派 CPU
  负担过大。
- 相机高度、俯仰角应与正式运行一致。每处额外采集一组开灯/关灯或白天/夜间照片，
  比单纯增加同一光照下的连拍更有价值。
- 避免大面积运动模糊；尽量包含门框、柜体边缘、木纹、固定设备等稳定结构。不要用
  临时摆放的箱子或人员作为地点真值。
- 参考位置直接来自巡检 `inspection_manifest.json` 的 `x_m/y_m/yaw_deg`，不另建一套
  地图坐标。

参考目录可以保持现有结构：

```text
inspections/ble_scene_17/<batch>/
  inspection_manifest.json
  01_...jpg
  01_..._camera2.jpg
```

## 建立预计算特征库

电脑环境先安装 `requirements-pi.txt` 中的 `opencv-python-headless`，然后运行：

```powershell
python -m tools.build_visual_reference_db `
  --references inspections/ble_scene_17 `
  --output inspections/ble_scene_17/visual_features
```

输出的 `index.json` 记录算法版本、图片绝对路径、特征文件和特征点数量；每张参考图的
关键点坐标、ORB 描述子和直方图保存为压缩 NPZ。运行时查询图只提取一次，参考图直接
加载 NPZ。缓存损坏时只回退重算对应图片，不会卡死定位状态机。

## 电脑离线测试

单图：

```powershell
python -m tools.test_visual_localization `
  --image test_images/query.jpg `
  --references inspections/ble_scene_17 `
  --feature-database inspections/ble_scene_17/visual_features/index.json `
  --top-k 3 `
  --debug-directory inspections/visual_localization/debug
```

文件夹会按文件名顺序作为连续停止帧，并输出 0.6/0.4 历史融合结果：

```powershell
python -m tools.test_visual_localization `
  --folder test_images/session_01 `
  --references inspections/ble_scene_17 `
  --top-k 3
```

重点观察：`score`、`inliers`、`good_matches`、Top-1/Top-2 `margin`、`reliable` 和
`fallback_to_global_active_localization`。调试目录中的连线图用于查看具体哪些局部结构
通过了 RANSAC。

## 当前限制

- 纯白墙和缺少角点的区域可能没有足够 ORB 特征；会 fallback，而不是猜一个位置。
- 重复走廊、相同柜门和相似地砖可能产生接近的 Top-1/Top-2；需要第二帧移动或固定
  语义地标消歧。
- 强逆光、夜间、镜面玻璃和大幅曝光变化会降低局部特征稳定性。
- 移动物体遮挡参考结构时，RANSAC 内点和 margin 会下降。
- ORB 是 CPU 友好的第一版。接口保留了候选列表与质量指标，后续可将特征提取器替换
  为轻量全局 embedding，而不改变雷达 fallback 和主动定位流程。
