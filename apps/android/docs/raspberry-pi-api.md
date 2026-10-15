# 手机端对接协议

App 默认把设置中的服务地址视为树莓派（或其转发服务）的 HTTP 根地址，例如：

```text
http://172.18.14.74:8770
```

对应实现为树莓派项目中的 `raspberry_pi.companion_service`。

## 健康检查

`GET /health`

```json
{"status":"ok"}
```

## 发送文本或语音识别结果

`POST /control`，`Content-Type: application/json; charset=utf-8`

```json
{
  "type": "set_instruction",
  "instruction": "捡起两个水瓶"
}
```

App 的语音模式先使用 Android 系统语音识别转成文字，再通过同一接口发送，便于树莓派统一处理。

服务器应返回：

```json
{"type":"ack","message":"instruction updated"}
```

急停、解除急停和任务重置分别使用：

```json
{"type":"emergency_stop","enabled":true}
{"type":"emergency_stop","enabled":false}
{"type":"reset"}
```

## 状态与回复

`GET /state`

App 前台每 1.5 秒轮询一次。它兼容项目中现有 3090 服务的字段，并允许树莓派增加回复文本：

```json
{
  "frame_index": 42,
  "grounding_status": "tracking bottle",
  "camera": {"connected": true, "fps": 9.8},
  "obstacles": {"status": "clear"},
  "lidar": {"status": "ready"},
  "emergency_stop": false,
  "robot_task": {
    "active": true,
    "task_type": "pick_multiple",
    "phase": "approaching",
    "phase_label": "正在接近目标",
    "completed": false
  },
  "robot_command": {
    "action": "forward",
    "reason": "target ahead"
  },
  "target_queue": [
    {"target_name": "第一个水瓶", "status": "done"},
    {"target_name": "第二个水瓶", "status": "active"}
  ],
  "task_plan": {
    "instruction": "去柜子旁找红色物块",
    "status": "running",
    "summary": "正在前往第二个观察点",
    "completed_steps": 2,
    "total_steps": 7,
    "map_url": "/task/map.png",
    "updated_at": 123456,
    "steps": [
      {"id":"step_01","title":"在永久地图中定位小车","detail":"定位已接受","status":"completed"},
      {"id":"step_02","title":"规划前往目标区域的安全路线","detail":"传统算法正在执行航点","status":"active"}
    ]
  },
  "services": {
    "process": {"kind":"object_search","running":true,"log":"service_runs/run.log"}
  },
  "assistant_message": "正在处理第二个水瓶",
  "reply_id": "reply-7"
}
```

回复正文可使用 `reply`、`assistant_message`、`message` 或 `text_response`；对象形式 `{"reply":{"text":"..."}}` 也可。建议每条新回复附带稳定且唯一的 `reply_id` 或 `message_id`，App 会据此避免重复显示和朗读。

任务进度优先按 `target_queue` 中 `done` / `completed` 的数量计算；没有队列时按 `robot_task.phase` 给出阶段性进度。

## 任务点地图

`GET /task/map.png` 返回当前寻物任务生成的地图。App 依据 `task_plan.updated_at` 刷新图片，不显示建图轨迹。

## 小车配置与服务控制

- `GET /config`：读取建图、标注和音量配置。
- `POST /config`：保存经过范围检查的配置对象。
- `POST /services/mapping/start`：按保存的参数启动停车稳定建图。
- `POST /services/annotation/start`：按保存的参数启动地图巡检和双摄像头标注。
- `POST /services/stop`：安全停止当前托管进程。

任意时刻只允许一个建图、标注或寻物进程占用底盘；服务端使用固定模块和参数白名单组装命令。

## 实时视频

默认地址为：

```text
GET /stream.mjpg
```

内容类型应为 `multipart/x-mixed-replace; boundary=frame`。也可以在 App 设置中单独填写其他 MJPEG 地址。当前实现与仓库既有 3090 服务的 `/stream.mjpg` 保持一致。

## 网络建议

- 手机与树莓派应连接同一个可信局域网。
- 当前允许局域网 HTTP 明文流量，正式跨公网部署时应改用 HTTPS、鉴权和证书校验。
- App 不直接连接 ESP32，也不解析 Vehicle Link V2 二进制帧；这些属于树莓派内部链路。
