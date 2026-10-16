# 视觉追踪逼近（pursuit）

语言指令 → 双目视觉锁定目标 → 追踪逼近 → 0.2m 左右自动停车。

**架构**：树莓派（双 USB 摄像头 + N10 雷达 + BLE→ESP32 底盘）与 3090
（Qwen3-VL-2B 视觉定位 + YOLO11n/CSRT 追踪 + SGBM 双目深度）之间通过
HTTP `/ground`（一次性识别）与 WebSocket `/pursuit_ws`（每帧上传，无背压）通信。

**2026-08-18 显示链路改造（显示帧率与感知帧率解耦）**：
- 网页不再轮询 `/frame.jpg`（感知帧率 + 双重编码），改为 `/stream.mjpg`
  MJPEG 流（multipart/x-mixed-replace，零重编码转发 Pi 上传的原始字节，
  帧率 = Pi 上传率 = 相机帧率）+ `/annotations` WebSocket（逐帧
  frame_result / pi_state / ground_result）在 canvas 上画框（EMA 平滑）。
- 网页 HUD 显示：浏览器显示 fps、采集 L/R fps、上传 fps、感知 ms、
  车状态、深度、bbox；断流 2.5s 自动重连（指数退避）。
- Pi 端：StitchWorker 后台线程拼接+编码一次（预览/上传共用字节）；
  `probe_camera_modes` 启动时两路同时实测档位（1280x720@30→15→960x540@30
  →640x480@30）；上传去背压、每帧发送（`--upload-fps` 变为可选上限，
  默认 0=每帧）；pi_state 增加 left_fps/right_fps/enc_fps/upload_fps/
  skip_fps/send_ms 字段。

**2026-08-20 鲁棒性与近距控制改造**：
- Qwen 锁定改为“左目识别 → 目标裁块作为右目参考”（或反向），并强制
  具体物体、完整入画、指令类别、左右名称和 y 坐标一致；未通过双目确认
  就返回未找到，禁止再把空画面的 floor/window 拼成追踪目标。
- 网页修正 `yolo_box` 字段层级，红色目标显示颜色分割的紧框，其余目标
  优先显示 YOLO 实测框；画面同时标出 left_picture/right_picture 掩码质心。
- 外观模板保留不可被近期误测覆盖的 Qwen 初始锚点，降低逐帧模板自污染。
- Pi 为每个感知结果记录接收时间和实际周期：结果年龄超过 0.45s 立即停车；
  感知过慢时限制两帧之间盲走距离不超过约 45mm。
- 进入示教慢速带或视觉近距带后使用“短脉冲前进—完全停车—等待新框”
  的视觉伺服，用时间占空比弥补电机无法稳定执行低速；示教模式任一路
  丢锁都停车并重搜，不再单目继续前冲。

```
树莓派 pursuit/run_visual_pursuit.py
  ├─ StereoCameraStream   双路摄像头自动探测 + 档位实测 + StitchWorker 拼接编码
  ├─ ServerClient         HTTP /ground + WS 帧流（每帧上传，motion 保序）
  ├─ PursuitController    TRACKING/SEARCHING 状态机（雷达 ICP 旋转重搜）
  └─ Esp32Client(BLE)     运动命令下发（默认 dry-run 只打印）

3090 3090/pursuit_server.py（端口 8021，8020 已被 arm_grasp_server 占用）
  ├─ POST /ground         Qwen 识别锁定目标（同步端点，线程池，不阻塞 WS）
  ├─ WS /pursuit_ws        逐帧 YOLO 匹配 → CSRT 兜底 → SGBM 深度
  ├─ GET /stream.mjpg      浏览器 MJPEG 流（零重编码，StreamHub 事件驱动）
  ├─ WS /annotations       浏览器标注广播（每客户端队列，latest-wins）
  └─ 3090/stereo_depth.py  双目校正 + 半分辨率 SGBM
```

## 3090 端启动

最短启动、网页观察和日志排障流程见仓库根目录
[`PURSUIT_RUNBOOK.md`](../PURSUIT_RUNBOOK.md)。推荐使用带日志归档的服务脚本：

```bash
cd ~/track
chmod +x pursuit_service.sh
./pursuit_service.sh restart --device cuda --no-depth
./pursuit_service.sh status
```

代码放 `~/track/3090/`（模型与虚拟环境复用 `~/3090` 的，不复制）。
**需确认 venv 装有** `python-multipart`（FastAPI Form/File）与
`websockets`（uvicorn 的 WS 支持，或装 `uvicorn[standard]`）：

```bash
~/3090/.venv/bin/pip install python-multipart websockets
~/3090/.venv/bin/python ~/track/3090/pursuit_server.py --port 8021
```

- 标定文件默认取脚本同目录 `configs/stereo_calibration.yaml`。
- Qwen 首次 `/ground` 调用才加载模型（数十秒），期间 WS 帧返回 busy 不积压。
- 可选参数：`--qwen-model`、`--yolo-model`、`--calibration`、`--device`。
- cv2 兼容 4.x/5.0：SGBM 用 HH 模式 + numDisparities 自适应回退
  （512→256→128）。cv2 5.0 下大 numDisparities 的左侧大面积无输出，
  由回退自动解决；4.x 首轮 512 即成功，无额外开销。3090 现装的是
  cv2 5.0.0.93，已在合成场景验证通过。
- 部署前自检：`~/3090/.venv/bin/python ~/track/3090/test_stereo_pipeline.py`
  （合成场景验证标定+校正+SGBM 深度换算，期望相对误差 <12%）、
  `~/3090/.venv/bin/python ~/track/3090/test_web_page.py`（HTTP/WS 接口层回归）。
  树莓派端逻辑自检：`~/dachuang/.venv/bin/python ~/dachuang/track/pursuit/test_pursuit_logic.py`。

## 测试网页

浏览器打开 `http://192.0.2.20:8021/`（树莓派端启动后才有画面）：

- **左右双摄像头画面**：MJPEG 实时显示（显示率跟随树莓派上传率），画面带目标框、
  深度标注与左右分界线；
- **指令输入框 + 重新识别按钮**：用输入的新指令在最新帧上重新
  Qwen 识别，结果经 WS 推给树莓派状态机——找到即切换追踪目标，
  未找到则按目标丢失进入旋转重搜；
- 状态行显示目标名/追踪状态/深度/bbox；
- 注意：若树莓派正处于 SEARCHING 的某一步旋转中，网页识别会在
  该步完成、树莓派自己的识别提交后生效。

## 树莓派端启动

代码放 `~/dachuang/track/`，在 track 根目录下运行
（`sys.path` 已自动加入 track 根，`car.protocol` 可被 import）：

```bash
cd ~/dachuang/track
~/dachuang/.venv/bin/python pursuit/run_visual_pursuit.py \
    --instruction "追踪那个蓝色瓶子" \
    --dry-run                    # 默认即 dry-run：只打印运动命令，不驱动实车
```

**实车模式必须先向使用者确认**，然后显式加 `--esp32`：

```bash
~/dachuang/.venv/bin/python pursuit/run_visual_pursuit.py \
    --instruction "追踪那个蓝色瓶子" --esp32
```

常用参数：`--stop-distance 0.20`（默认 0.2m）、`--upload-fps`、
`--left-camera N` / `--right-camera N`（排障用）；2026-09-25 实测
2026-09-25 现场确认 `/dev/video0` 是物理右目、`/dev/video2` 是物理左目，追踪默认交换后再上传；网页的“交换左右”只影响网页显示，不改变底层追踪顺序。
再次换线后可用 `--swap-cameras` 临时覆盖；
`--no-lidar`（无雷达退化模式）、
`--log-dir 路径`（默认 `logs/pursuit`）。`--log-csv` 仅作为兼容的另存路径。

每次运行自动建立独立目录，不再覆盖上一次日志。目录包含 `run.json`、
`console.log` 和逐秒的 `telemetry.csv`；`logs/pursuit/LATEST.txt` 指向最近一次。
快速汇总：

```bash
cd ~/dachuang/track
~/dachuang/.venv/bin/python pursuit/summarize_run.py
```

## 摄像头自动探测

新代码**不写死摄像头编号**：优先 `/dev/v4l/by-path`、`by-id` 稳定节点
（每物理摄像头取 `-video-index0`），解析出整数编号后以 V4L2 打开，
只认能按 MJPG 1280x720 实际取到帧的设备；稳定节点不足时回退遍历
`/dev/video0..15`。启动日志会打印探测结果：

```
[Camera probe] /dev/video0 OK (/dev/v4l/by-path/...-video-index0), frame=1280x720
```

## 关键配置（pursuit/pursuit_config.py）

- `PursuitConfig.stop_distance_m = 0.20`：用户要求的停车距离；
  双目深度与雷达前向距离取最小值交叉确认，连续 3 帧进入 STOPPED。
- `PursuitConfig.search_rotation_steps_deg = (45, -90, 135, -180, 90, 0)`：
  丢失目标后停车原地旋转重搜，每步用雷达相邻圈 ICP 确认转角，
  到位即重新提交 Qwen 识别；转满仍未找到 → GIVE_UP。
- `StereoConfig.yaw_compensation_px = 0.0`：相机 yaw 标定（~4.43°≈97px）
  的符号与数值待实车直行对准验证后再填入。
- 安全底线：雷达前向 <0.12m 立即硬停；所有运动命令带短 TTL
  （近端 400ms/远端 600ms），ESP32 端另有 800ms 无命令急停兜底。

近端丢失找回：先停并检查最新双目画面。现场确认后向雷达被车身
遮挡，默认 `recovery_rear_sensor_usable=False`，不读取后向扇区，
不把车身固定回波当成环境障碍。只有在最近 1.6 秒内确实下发过
前进命令、双目连续帧显示目标正在接近、且雷达扫描仍新鲜时，
才允许最多两次短后退。每次都发送紧急停车、等待新视觉帧再决定；
按固件 650mm/s² 加速度计算的总行程预算为 10cm。扫描过期、
没有新画面、缺少近期前进证据或预算已用尽时，
立即停车。这个预算是运动模型上限，不是实测位移；首次实车验证应
有人在场观察。相关参数在 `PursuitConfig.recovery_swept_path_*`，
每一步的原因写进本次 `telemetry.csv` 和 `console.log`。

## 二维停车区与四维示教凸包微调

二维掩码仅是追踪阶段的停车/交接区域，不证明双目联合坐标可抓取。
`final_work/pi/pursuit_session.py` 恢复抓取服务的 640×480 双目画面后，
调用 `pursuit/hull_micro_adjust.py`，按示教模型返回的最近四维凸包点
进行停车—观察—短脉冲调整。模型输入仍保持示教时 video0/video2 顺序，
不随网页左右显示标签交换。

近端微调分为三档（均不是机械臂抓取容差）：

- 常规档：四维距离 60–180px。
- 贴近档：距离 ≤60px。
- 边界试探档：180–240px；
  不再因 180.4px 略超原入口而直接拒绝尝试。

直线微调复用此前能成功起步的逼近参数：
`PursuitConfig.visual_pulse_mm_s`（当前 405mm/s）和
`visual_pulse_near_s`（当前 0.80s，上限也为 0.80s）。
停车间隔复用 `visual_pulse_rest_s`（0.45s），之后等新的双目识别结果。
不采用远距离的 2.00s 卡住升级。固件速度从零爬升，过短的 0.40s
脉冲曾无法起步；后续按视觉反馈缩短时，直线时长不低于当前约
0.68s 起步下限（405/650 + 0.06s），原地转向仍为 0.40s。
这个下限是起步参数假设，仍需实车验证，不是厘米级位移保证。

追踪逼近阶段保留 405mm/s、首次 0.80s；只有连续两步双目确认
目标几乎不动才每档增加 0.05s，远处最多 0.95s、近处最多 0.85s。
双目丢失、身份不确定或链路异常不能作为静摩擦证据；双丢先停车
重新识别，仍找不到才进行有累计时间上限的搜索。凸包微调的
0.80s 上限与上述追踪逼近档位不同，不混用。

共享 BLE 转向接口新增背景光流早停：两路整体运动均达到 8px、
或任何一路达到 12px 就请求 STOP；画面冻结、特征不足或匹配
不可靠也停车。使用 RANSAC 抑制局部目标/手部运动的影响，
但不能保证完全区分人为移动与底盘运动。阈值是 640×480 图像
像素，不是转角或行驶距离；惯性及通讯延迟仍可能造成超调。
人工移动目标时，该段目标中心变化不能用于标定车辆响应。

这些像素是模型归一化四维残差的等效单位，不能换算成实际行驶厘米。
每步停车后重新识别双目目标并检查凸包距离。改善至少 2px 就继续
微调，直到模型明确判为凸包内。方向选择联合使用两路 X/Y（Y 按
640/480 加权，与模型的四维距离一致），不再用水平视差单独覆盖
纵向误差。初次动作采用内八字几何先验，之后由本会话实测四维中心
变化估计局部响应；该响应不跨目标、会话或相机重启复用。
后续选择在起步时间下限仍预计改善 ≥2px 的动作；如果最小脉冲已会
过冲，就不能持续重复或借放宽凸包启动机械臂。

首次可见移动却未改善时，若误差增大 ≤20px、位移 ≤100px、视差
方向正常且目标仍在双目内，允许一次停车稳定性复核（漂移 ≤15px）
后重规划的机会。禁止重复失败方向，优先利用实测响应选择能改善的
纠正动作。纠正后改善则继续微调；第二次不改善、大幅恶化、目标
丢失、稳定性复核失败或预算不足则停止，不无限来回摆动。
没有移动时只增加
0.10s 脉冲时间，不提高速度、不超过 0.80s；已经用到上限仍没有
视觉进展就停止。后续脉冲只按实测双目中心变化、视差和四维凸包
误差缩短，纠正动作也不低于直线起步下限。默认最多 5 步，累计
下发脉冲时间 ≤3.0s。
目标丢失、相机顺序不符、雷达过期/障碍、实测视差方向异常和急停
仍会阻止继续移动。凸包微调也不读取被遮挡的后向扇区，只检查
前向有效扇区（直行 0°，转向增加 ±45°）。这不能证明后方没有
动态障碍，实车短退时应有人观察。

追踪会话向凸包微调交接时，用本次子进程的日志核对：实车模式、
同一目标、正常掩码停车，最后 3 秒无转向/倒车、双目持续有目标、
确有前进命令且两路目标中心都向近端移动。仅这样的最新路径
可获得 90 秒有效的原路短退凭据；旧日志、单目丢失、转向或失败
会话不能授权盲退。四维微调的短退改为最多两步，每步 ≤0.80s，
累计通电 ≤1.60s，并且每步之间必须停车复核双目四维凸包。
不再按理想速度爬升把“0.40s 未起步”扣成约 8cm 实际路程，
也不使用前向雷达位移/深度估计决定步长。**该阶段不保证 10cm
物理上限**：这是受限视觉步进而非米制里程控制；实车应留出退路
并有人观察。目标丢失或凭据过期立即结束；视觉误差不改善仅允许
上述一次受限纠正。转向后原直线路径凭据失效，不继续无凭据后退。

共享 BLE 的 `chassis-pulse` 端点接收 `speed_mm_s`，按固件百分比
编码，因此修改逼近速度也会传递到微调，不再在底层固定为 S074。
当前端点限制直线速度在 100–405mm/s，超限配置会拒绝执行而非
偷偷提速。每次日志记录速度、起步窗口、视觉误差改善量和短退
累计通电时间。

只有模型明确返回 `inside_teaching_hull=true`，微调才成功。
试探带不扩大示教凸包，也不授权机械臂在凸包外启动。
参数在 `hull_micro_adjust.py`；`--trial-band-px 0` 可恢复严格入口。
日志位于 `logs/hull_align/*.jsonl`，包含分档、计划脉冲、累计脉冲时间、
前后凸包距离改善量、视差变化、局部四维响应、决策来源、纠正前
稳定性复核、继续微调/完成事件及最终停止原因。

仅观察、不移动的检查命令：

```bash
cd ~/dachuang/track
~/dachuang/.venv/bin/python pursuit/hull_micro_adjust.py --target "红色方块"
```

## 状态机

```
INIT → TRACKING（持续逼近）→ STOPPED（成功，≤0.2m）
        ├─ 目标丢失 1.2s → SEARCHING（旋转重搜）
        └─ SEARCHING 转满 / 雷达硬停 → GIVE_UP
```

## 依赖

树莓派 `pursuit/requirements.txt`（numpy、opencv-contrib-python、
websocket-client、pyserial、bleak）安装进 `~/dachuang/.venv`。
