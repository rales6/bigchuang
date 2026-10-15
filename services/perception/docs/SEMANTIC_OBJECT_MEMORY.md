# 语义物体清单、补充标注与任务复用

本次巡检结果和相关代码已保存在 `backups/semantic_inspection_20260917_195037.zip`，原备份不修改。巡检实际投影视野覆盖约 88.4%，不代表已经拍到全部墙面设施。

`3090/scripts/run_semantic_object_inventory.py` 有三个模式：

- `--replay panorama_semantics.json`：仅整理旧 Qwen 结果，不进行新推理。当前已整理全部 33 组照片记录、221 个观测，得到 91 个粗粒度候选聚类，绝不是场景里存在 91 个独立物体。
- `--ground-url http://127.0.0.1:8010/agent/ground`：复用运行中的模型，按类别补扫。默认插座、开关、标牌、灯具。整图加上部 45% 细节裁剪，两路照片分别查询；加 `--baseline panorama_semantics.json` 保留旧大物体。
- `--model-path /path/to/Qwen3-VL`：独立本地模型做全可见物体识别和上部细节补扫。没有本地模型时不运行；3090 已有服务时避免加载第二份模型占用显存。

均需指定 `--inspection`、`--map`、`--map-yaml`、`--output`；输出不能放在源巡检或原地图目录内。支持 `--limit 1` 小样本、`--previous` 保留上一版记忆、`--reviews` 坐标确认；成功检查点会缓存，失败项单独记录并允许续跑。

共享服务示例（未来在 3090 执行，不是本轮部署命令）：

```bash
python scripts/run_semantic_object_inventory.py \
  --inspection /path/to/20260917_190706 \
  --map /path/to/ble_scene_17.pgm --map-yaml /path/to/ble_scene_17.yaml \
  --ground-url http://127.0.0.1:8010/agent/ground \
  --baseline /path/to/20260917_190706/panorama_semantics.json \
  --queries 插座,开关,标牌,灯具 --output /path/to/new_inventory --limit 1
```

先前已在 3090 复用服务实测一组插座照片、四次请求：成功识别紧致插座框。证据位于 `semantic_maps/ble_scene_17_20260917_190706/live_qwen_probe`。该测试不是全套照片重新识别；本轮只在电脑本地完成整理和测试，没有新增服务器/树莓派操作。

## 结果及可用性

输出 `semantic_object_memory.json`（schema `semantic_object_memory.v1`）、`object_inventory.csv` 和 `inventory_report.json`。保存物体类别/别名、ID、照片框、站点、拍照位姿、位置估计方式、地图哈希、玻璃外状态和变化待核查记录。

识别名称和世界位置是两项独立证据。未标定相机或未确认世界坐标时，只记录近似搜索区域。默认 `navigation_allowed`、`localization_allowed`、`grasp_allowed` 都是 false。看到桌子不能把整段黑色墙全部变成桌子；插座和开关共用旧框时不能当作两个已确认物体。

补充地图位于 `semantic_maps/ble_scene_17_20260917_190706/complete_objects/supplemented_object_map.png`，15 个区域标签，文本无重叠。包含两个可能重复的插座候选、出口标牌、玻璃外灯具、柜面标记。`supplemented_evidence.json` 还保留未确认开关/立柱，以及玻璃门的门框/把手部件关系。第二路相机安装方向尚未标定，所以这些位置仍须实测确认，不能用它们反向“校正”小车。

## 如何确认地标以供定位

reviews JSON 顶层填实际 `map_sha256`；objects 中用输出的真实 ID，提供 `reviewer`、`reviewed_at`、`verified_map_xy` 和 `mobility: fixed`；明确设置 `localization_allowed: true`。必须是人工/实测确认，不能由模型自批。

若还需导航，另提供旧图自由区域的 `verified_approach_xy` 和 `navigation_allowed: true`。这一坐标检查只验证单点；任务执行器仍须检查完整底盘和实时障碍。物体中心不能充当停车点。玻璃外物体不能批准本场景导航/定位，可移动物体不能批准固定地标定位。

旧记忆本轮未观察到时保留并添加 `unobserved_not_proven_absent` 疑似变化标记，不删除旧物体、不覆盖原占据地图。后续到这里再次视觉检查；确有变动才追加变化证据。

任务专用照片定位设计和接口见 `raspberry_pi/docs/task_semantic_localization.md`；标注巡检逻辑不替换。当前清单已验证定位锚点数为 0，先完成固定地标坐标/相机校准，才可启用实车强纠错。

## 2026-09-18 完整重标注部署记录

后续已实际完成全部 33 组照片的新一轮 grounding：66 个阶段、1122 次请求，无推理失败。记录见 `semantic_maps/qwen_reannotation_20260918_142739`；原始 735 个观测经低置信过滤、重复框合并和类别冲突标记后，保留 303 个图像区域观测。排除 359 个低/零置信观测，45 个区域有类别冲突。

应读取 `results/quality_filtered/semantic_object_memory.json`，而不是未经质量过滤的原始聚类。过滤后仍有 150 个候选聚类，不是 150 件确定物体。世界位置仍需校准/核实，没有强定位锚点。

树莓派独立结果目录：`/home/dachuang/dachuang/car1.0/semantic_maps/qwen_reannotation_20260918_142739`。独立任务定位模块已部署并在 `.venv-pi` 下验证导入、依赖和实际语义文件加载；任务照片 HTTP 接口改用标准库，不需要安装 requests。未启动车辆运动、未替换原标注巡检定位流程、未更改原 PGM/YAML。
