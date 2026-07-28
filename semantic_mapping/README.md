# Semantic Mapping Prototype

这个文件夹是一个独立实验版，不修改 `car/` 里的 ESP32 小车代码，也不修改原来的 `raspberry_pi/` 建图代码。

目标是把现有雷达 SLAM 与 Qwen 视觉语义结合起来：

- 雷达继续负责连续建图、局部避障和位姿估计。
- 摄像头提供当前场景图像。
- 3090 上的 Qwen 低频选择自然参考点和语义物体。
- 树莓派把这些参考点保存到语义地图 JSON 中，后续可以按“去垃圾桶旁边”“回到门口附近”这类描述查询目标位置。

## 1. 设备分工

### 树莓派运行

`semantic_mapping/raspberry_pi/run_semantic_mapping.py`

作用：

- 读取 N10 雷达。
- 复用原项目 `raspberry_pi.mapping.LidarSlam` 做占据栅格建图。
- 读取摄像头画面。
- 在启动、转弯后、雷达匹配变差、参考点过少、周期巡检时，向 3090 请求 Qwen 语义参考点。
- 保存普通地图文件和语义地图 JSON。

### 3090运行

`semantic_mapping/3090/run_qwen_landmark_server.py`

作用：

- 加载本地 Qwen3-VL 模型。
- 接收树莓派上传的摄像头图、局部雷达图和当前 SLAM 状态。
- 返回适合长期记忆的自然参考点，例如墙角、门框、固定柜子、固定垃圾桶、柱子，以及可用于任务导航的语义物体。

### ESP32不改

ESP32 仍只负责执行数值命令，比如速度、停止、机械臂关节。语义地图不会进入 ESP32。

## 2. 先启动 3090 Qwen 语义服务

在 3090 上：

```bash
cd ~/bigchuang
source 3090/.venv/bin/activate

python semantic_mapping/3090/run_qwen_landmark_server.py \
  --model-path 3090/models/qwen/Qwen3-VL-2B-Instruct \
  --host 0.0.0.0 \
  --port 8010 \
  --device cuda
```

检查：

```bash
curl --noproxy '*' http://127.0.0.1:8010/health
```

如果只想先测试树莓派流程、不加载模型，可以加：

```bash
--mock
```

## 3. 树莓派运行语义建图

在树莓派上：

```bash
cd ~/bigchuang
source .venv/bin/activate

python semantic_mapping/raspberry_pi/run_semantic_mapping.py \
  --qwen-url http://192.168.55.33:8010/landmarks \
  --camera 0 \
  --lidar-port /dev/ttyUSB0 \
  --esp-port /dev/serial0 \
  --output maps/semantic_room
```

如果暂时不连接 ESP32，只采集雷达和视觉：

```bash
python semantic_mapping/raspberry_pi/run_semantic_mapping.py \
  --qwen-url http://192.168.55.33:8010/landmarks \
  --camera 0 \
  --lidar-port /dev/ttyUSB0 \
  --no-esp32 \
  --output maps/semantic_room
```

输出文件：

- `maps/semantic_room.pgm`
- `maps/semantic_room.yaml`
- `maps/semantic_room.png`
- `maps/semantic_room_trajectory.csv`
- `maps/semantic_room_semantic.json`

## 4. 查询语义地图

建图后可以查询“垃圾桶”“门口”“柜子”等目标：

```bash
python semantic_mapping/raspberry_pi/query_semantic_map.py \
  --map maps/semantic_room_semantic.json \
  --query 垃圾桶
```

输出中会给出候选地标的地图坐标、稳定度和说明。后续可以把这个坐标接入小车导航。

## 5. 什么时候调用 Qwen

脚本默认不会每帧调用 Qwen，而是由事件触发：

- 启动阶段。
- 雷达 ICP 连续异常。
- 转弯角度累计较大。
- 参考点数量不足。
- 每隔一段时间低频巡检。

这样 Qwen 负责“选择和解释参考点”，雷达负责“连续建图和几何拼接”。

## 6. 当前限制

这个版本不会把 Qwen 输出直接当成精确坐标，也不会强行重置 SLAM 位姿。原因是自然环境没有人工标记，Qwen 的语义结果适合做参考，不适合直接做厘米级定位。

当前版本先完成：

- 语义参考点发现。
- 语义地图 JSON 记忆。
- 雷达几何特征和视觉语义的关联。
- 后续导航目标查询。

后续可以继续增强：

- 用稳定参考点修正 yaw。
- 加入回环检测。
- 将地标匹配接入 pose graph。
- 使用双目粗深度估计给视觉地标更准确的地图坐标。
