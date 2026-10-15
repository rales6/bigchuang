# 自然语言寻物与已知地图路径规划（阶段一）

本阶段只生成经Qwen审查的寻物契约和A*搜索路线，不直接发送电机或机械臂命令。

## 决策原则

- 当前永久地图之外的任务直接拒绝，例如“去马路上找红色物体”。
- 地标存在时围绕地标生成多个可达观察点，例如“去柜子旁找红色瓶子”。
- 地标不存在且没有方位时追问；用户补充“在地图右下角”后按右下区域生成路线。
- Qwen只解释语义，不能输出坐标；树莓派根据PGM/YAML和A*确定路线。
- 新用户指令立即抢占空闲语义标注；已有用户任务也由更新的指令替换。
- 空闲标注使用去重键，同一地图版本、目标和视角不重复写入。

## YOLO与Qwen视觉回退

当前`yolo11n-seg.pt`是COCO 80类。`bottle`、`cup`、`book`、`chair`等已知类别优先使用
YOLO几何框；`block/积木/小物块`等YOLO不存在的类别不再直接拒绝，而是在小车先到达指定
地标或区域附近并停车后，切换到Qwen视觉定位。

无论使用哪种检测模式，都不能让模型一句“找到了”直接结束任务。Qwen回退必须同时满足：
Qwen来源、置信度不低于0.55、有效紧致边界框、类别严格相符；带颜色要求时还要通过目标框
像素颜色校验。当前视角没有可靠结果时，小车继续前往目标附近的下一个可达观察点；所有附近
观察点耗尽后才停车并反馈“未找到”。

完整顺序是：雷达定位 → A*到达目标附近 → 停车稳定 → 拍照并调用YOLO/Qwen → 未找到则
在目标附近换观察位继续搜索。赶往目标区域途中不会进行目标成功判定。

## 3090启动

```bash
mkdir -p ~/car_runtime
# 从树莓派复制最新派生语义地图到 ~/car_runtime/ble_scene_17_semantics_qwen.json
cd ~/3090
chmod +x start_search_task_api.sh
./start_search_task_api.sh
```

提交指令：

```bash
curl -s http://127.0.0.1:8030/command \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"去柜子边上找一个红色瓶子"}' | python -m json.tool
```

如果系统追问位置，下一次直接提交补充信息：

```bash
curl -s http://127.0.0.1:8030/command \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"簸箕在地图的右下角"}' | python -m json.tool
```

## 树莓派离线路线验证

将3090返回的`decision`保存为`search_decision.json`后执行：

```bash
cd ~/dachuang/car1.0
source .venv-pi/bin/activate
python3 -m raspberry_pi.search_route_planner \
  --map maps/ble_scene_17.pgm \
  --map-yaml maps/ble_scene_17.yaml \
  --decision search_decision.json \
  --start-x 0.0 \
  --start-y 0.0 \
  --output search_route.json \
  --render search_route.png
```

`--start-x/--start-y`必须来自本次任务的真实雷达定位。规划器不会读取建图轨迹作为起点，
也不会向电机发送任何命令。
