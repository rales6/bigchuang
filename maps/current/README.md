# 当前地图与定位参考

当前实车发布来源：树莓派 `/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/`。

- `ble_scene_17.pgm`、`ble_scene_17.yaml`：占据图、分辨率与坐标。
- `semantic_runtime.json`：当前语义对象、区域、关系和使用约束。
- `semantic_localization_landmarks.json`：视觉定位的永久地标。
- `semantic_relationship_map_20260919.png`：人工查看的关系语义图。
- `reviewed_layout_20260917.json` 和同目录 Qwen/组合 JSON：当前发布构建所需的标注证据，不是额外的可切换导航地图。
- `references/`：当前定位加载的 20 个参考地点、40 张图片及索引；来源 `inspections/ble_scene_17/20260918_201704/`，不是把所有历史拍摄收回。
- `references/inspection_manifest.remote.json`：远端原始索引；`inspection_manifest.json` 将照片路径本地化。
- `manifest.json`：远端发布时的原始哈希表。表中旧 `.py` 脚本已收在 `robot/scene_runtime/`，地图目录只放数据。

`occupancy_success_20260917_190706.*` 是当前地图的来源审计副本，PGM 与当前地图哈希相同，不是另一套实验地图。其他日期的地图和拍摄结果仍在树莓派，旧电脑目录在仓库外归档。

电脑分析请显式传入本目录的路径；设备上的地图、服务配置与绝对路径此次未改。离线仿真已改为读取本目录，不会覆盖它。
