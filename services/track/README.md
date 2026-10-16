# final_work：双设备机器人 Agent 网页

## 当前能做什么

- 3090 提供一个 GPT 风格单网页：聊天历史、Qwen 结构化计划、任务步骤、地图/定位、双目视频和日志。
- `final_work.gpu.qwen_hub` 保留原 8010 全部语义地图接口，并让 Agent 规划与语义标注共享同一份 Qwen 模型和串行锁，避免重复占用显存。
- final_work 将 Qwen 常驻副本限制为两份：8010（规划/地图/追踪首次定位共享）与 8020（抓取专用）。8011 停止，8021 只保留 YOLO/CSRT 并通过 HTTP 调用 8010。
- 会话、消息、Qwen 计划与执行状态持久化到 `final_work/data/history.sqlite3`，重启后仍可读取。
- 地图与双目视觉通过右下角悬浮工具唤起，不挤占聊天工作区；地图支持拖动、滚轮/按钮缩放、复位，并在实时位姿可用时自动把小车置于视图中央。
- 地图弹窗可切换雷达底图、Qwen 语义标注、稳定发布图层和当前导航路线；语义实体名称直接来自只读发布 JSON。
- 设置位于左侧栏底部，使用独立设置面板，可切换简体中文/English 和本地 Qwen、Qwen API、DeepSeek API、GPT API。没有“规则规划”用户选项。
- 树莓派桥接层复用现有程序，不接收任意 shell：
  - `raspberry_pi.companion_service:8770`：建图与安全停止；
  - `pursuit/run_visual_pursuit.py` + 3090 `:8021`：识别目标并持续追踪靠近；
  - `/home/robot/grab_visual/web_app.py` + 3090 `:8020`：双目预览、视觉伺服抓取、释放；
  - 雷达实时地图 `:8766`：地图 PNG、位姿与定位质量。
- Qwen 只产生高层、白名单化的结构计划；低层运动仍由现有确定性状态机与雷达保护完成。

## 地图标注与导航接入

- 正式数据源固定为只读目录 `/home/robot/dachuang/car1.0/ble_scene_17_relationship_20260919/`。运行时语义读取 `semantic_runtime.json`，网页默认显示 `semantic_relationship_map_20260919.png`，定位/A* 使用同目录且哈希一致的 `ble_scene_17.pgm/.yaml`。
- `semantic_localization_landmarks.json` 只供既有视觉定位模块消歧；正式发布记录中已验证定位地标数仍为 0，因此系统不会把未审核候选当成强定位或抓取坐标。
- `semantic_combined_20260919.*` 仅作为旧自动投影的追溯材料，不再作为网页默认层或当前布局。
- 3090 的共享 Qwen Hub 在原 8010 服务上增加 `/mission/command`，复用同一份模型把自然语言转换为目标、地图锚点和搜索区域，不启动 8011 或第三份 Qwen。
- Pi 的 `final_work.pi.map_navigation_adapter` 以原 `search_object.sh` 的同一组保守运动参数调用冻结的 `raspberry_pi.object_search_mission`，并显式传入上述正式地图、YAML 和定位语义文件。安全模式使用 `--simulate`，实车模式仍须通过安全令牌。定位、A*、多观察点、雷达避障和停车视觉核验仍由 `car1.0` 原模块负责。
- 观察点采用“两级确认”：冻结规划器先依据占据栅格净空、目标可见射线和 A* 可达性生成候选；共享 Qwen 只能对候选编号进行语义排序，禁止生成坐标。排序后 `final_work` 会重新计算每一段 A* 路径，再交回冻结执行器。Qwen 不可用或输出越界时保留原安全顺序。
- “在 A 附近找到并夹取 B”会明确拆分为搜索锚点 A、视觉/抓取目标 B 和 `pick` 意图，避免把 A 当成追踪目标或把 B 错当放置地点。
- 启动巡航或导航前，如任务槽仅被 `camera_preview` 占用，3090 会先自动停止纯预览；不会抢占追踪、导航、建图或抓取任务。
- `search_runs/live_status.json` 与 `search_runs/live_map.png` 被接入网页，建图地图服务未运行时仍可显示最近定位、导航阶段和路线图。
- 因 3090 所在网段无法主动路由到 Pi，Pi 使用专用受限 SSH 密钥维护 `3090 127.0.0.1:18780 → Pi 127.0.0.1:8780` 反向隧道；该端口不对局域网开放，隧道断开后每 2 秒自动重连。
- 当前冻结版本完整支持“去某个语义区域附近寻找/观察指定物体”。单纯前往任意未标注坐标、以及找到目标后的真实抓取，仍会保守暂停；不会把 Qwen 候选框中心当成导航或抓取坐标。

## 模型 API 设置

网页左侧栏底部打开“设置”，选择 API 服务后填写该账号可用的模型 ID 和密钥。服务地址固定为各服务的 OpenAI 兼容端点，不能由网页任意填写，避免请求被转发到内网未知地址。

Qwen、通用 DeepSeek 与 GPT 兼容项可使用服务器端密钥。单独的 `DeepSeek V4.1 Flash API` 项采用更严格的使用者设备模式：密钥只保存在当前浏览器，完全不发送到 3090；浏览器直连 DeepSeek，必要时经过使用者设备本机的回环桥。外部模型只提取 `intent/target/destination`，最终动作步骤仍由 3090 上的白名单编译器产生。

## 已接入接口

- `GET /map/layers`：列出固定白名单地图图层；
- `GET /map/layer.png?id=...`：读取一个已登记图层；
- `GET /semantic-map`：读取稳定发布或兼容语义数据；
- `POST /interfaces/observation-plan`：自然语言规划多观察点寻物，并启动仿真/实车任务；
- `POST /interfaces/navigation-plan`：根据地图锚点或粗略区域规划导航搜索，并启动仿真/实车任务；
- 3090 `POST :8010/mission/command`：共享 Qwen 的冻结寻物契约兼容入口。

这些接口只接受结构化字段，不接受 shell、路径或任意 URL。`car1.0` 的建图、标注、定位、路径规划和运动文件保持只读。

## 启动

先启动树莓派桥接（默认安全模式，不允许真实建图/抓取，视觉追踪自动加 `--dry-run`）：

```bash
cd /home/robot/dachuang/track
bash final_work/run_pi_background.sh
```

3090 首次部署先把原 8010 语义服务切换为兼容的共享 Qwen Hub，然后启动网页：

```bash
cd /home/gpu/track
bash final_work/run_qwen_hub_background.sh
bash final_work/enforce_two_qwen.sh
bash final_work/run_3090_background.sh
```

浏览器打开：`http://192.0.2.20:8030/`

现有感知服务应保持：

- `http://192.0.2.20:8010/`：共享 Qwen 语义地图服务；`/agent/plan` 为规划接口；
- `8011`：final_work 模式下停止，避免重复加载 Qwen；
- `http://192.0.2.20:8020/health`：机械臂抓取服务；
- `http://192.0.2.20:8021/`：视觉追踪服务。

## 实车模式

只有清空小车和机械臂周围环境后才启用。两端必须使用同一随机令牌：

```bash
# 树莓派
cd /home/robot/dachuang/track
bash final_work/stop_pi.sh
export FINAL_WORK_ENABLE_HARDWARE=1
export FINAL_WORK_HARDWARE_TOKEN='替换为随机长字符串'
bash final_work/run_pi_background.sh
```

```bash
# 3090
cd /home/gpu/track
bash final_work/stop_3090.sh
export FINAL_WORK_HARDWARE_TOKEN='同一个随机长字符串'
bash final_work/run_3090_background.sh
```

网页即使在实车模式也不会自动连续执行整个 Qwen 计划：操作员先确认计划，再逐步执行。急停无需硬件令牌。

## 日志与历史

- Pi 桥接和受管任务日志：`/home/robot/dachuang/track/final_work/logs/`
- 视觉追踪每次任务详细日志：`/home/robot/dachuang/track/logs/pursuit/`
- Pi 抓取日志：`/home/robot/grab_visual/logs/`
- 3090 抓取日志：`/home/gpu/grab_visual/logs/`
- 网页/Qwen 服务日志：启动命令的标准输出，部署脚本保存到 `/home/gpu/track/final_work/logs/`
- 聊天与计划：`/home/gpu/track/final_work/data/history.sqlite3`（SQLite WAL 模式）

日志页读取当前 Pi 受管进程末尾约 24 KB；完整排障仍以对应目录中的时间戳文件为准。

## 端口

| 设备 | 端口 | 用途 |
|---|---:|---|
| 树莓派 | 8770 | 既有 companion service |
| 树莓派 | 8766 | 建图运行时地图服务 |
| 树莓派 | 8780 | final_work 安全桥接 |
| 树莓派 | 8080 | 按需启动的双目/抓取客户端 |
| 3090（仅回环） | 18780 | Pi 8780 的受限 SSH 反向隧道 |
| 3090 | 8030 | final_work 网页与 Agent API |

摄像头是独占资源：启动视觉追踪时桥接层会停止双目预览进程；抓取前会重新启动双目/抓取客户端。不要同时手工启动另一份相机程序。

停止服务：

```bash
# 树莓派
cd /home/robot/dachuang/track && bash final_work/stop_pi.sh

# 3090
cd /home/gpu/track && bash final_work/stop_3090.sh
```

如需恢复未经包装的原 8010 服务：

```bash
cd /home/gpu/track && bash final_work/restore_original_qwen.sh
```
# 用户设备上的 DeepSeek Flash

设置页可选择 `DeepSeek V4.1 Flash API`（模型 ID 默认为 `deepseek-flash`）。API 密钥只保存在当前浏览器的 `localStorage`，不会发送到 3090 的设置接口，也不会写入服务器数据库、文件或日志。浏览器优先直接请求 DeepSeek；若浏览器阻止跨域访问，在打开网页的使用者设备上运行：

```powershell
python final_work/tools/deepseek_local_bridge.py
```

本机桥只监听 `127.0.0.1:8765`，不保存或打印密钥，并只接受来自 `http://192.0.2.20:8030` 的网页请求。3090 只接收 DeepSeek 返回的意图 JSON，并再次进行白名单校验和确定性计划编译。

# 双目左右校准

双目视觉弹窗提供“AI 校准左右”和“手动交换”。本车两只摄像头为内八字、略向中央汇聚安装，几何判断使用经操作员现场确认的内八字视差约定，不能套用平行/已校正双目的视差符号。AI 校准复用 8010 上现有 Qwen，并结合双目特征视差；只有证据一致且置信度足够时才自动应用，冲突或低纹理场景保持原映射。保存的只有左右映射、置信度和原因，不保存校准图像。
