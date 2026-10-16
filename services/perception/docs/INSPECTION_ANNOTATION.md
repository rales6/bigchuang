# 3090 Qwen 巡检语义标注

该流程读取树莓派 `map_camera_inspection.py` 保存的 `inspection_manifest.json` 及单/双摄照片，
使用 Qwen3-VL 为每个墙面或地图黑色区域生成语义类别、中文说明、可见特征、置信度和逐视角目标框。

输出保存在所选巡检批次目录：

- `qwen_annotations.json`：完整标注、模型原始回答和断点信息；
- `qwen_annotations.csv`：便于人工查看的表格；
- `qwen_annotated_images/`：带目标框的图片；
- `semantic_map_qwen.json`：从原语义地图派生的新语义地图。

程序不会覆盖 `--semantic-map` 指定的树莓派原始语义文件。置信度不足、目标不可见、疑似临时物体
或模型要求复核时，只写入 `qwen_candidate_label`，不提升为正式 `semantic_label`。

## 执行

先将巡检结果和原语义文件从树莓派同步到 3090：

```bash
mkdir -p ~/car_inspections/ble_scene_17
rsync -av dachuang@192.0.2.11:~/dachuang/car1.0/inspections/ble_scene_17/ ~/car_inspections/ble_scene_17/
rsync -av dachuang@192.0.2.11:~/dachuang/car1.0/maps/ble_scene_17_semantics.json ~/car_inspections/
```

在 3090 上运行：

```bash
cd ~/3090
chmod +x run_inspection_annotation.sh
./run_inspection_annotation.sh ~/car_inspections/ble_scene_17 \
  --semantic-map ~/car_inspections/ble_scene_17_semantics.json
```

传入巡检根目录时，程序自动选择修改时间最新的 `inspection_manifest.json`。每完成一个目标就保存
断点；重复执行会复用输入照片未变化的标注。使用 `--force` 才会全部重跑。

首次部署若模型权重尚未下载：

```bash
cd ~/3090
source .venv/bin/activate
pip install -e .
pip install -r requirements/qwen.txt
python scripts/download_models.py --qwen --source modelscope
```
