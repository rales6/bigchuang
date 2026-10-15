# 视觉追踪启动与日志操作手册

下面的顺序默认先启动 3090 视觉服务，再启动树莓派。测试时先用
`--dry-run`，确认网页识别框、左右掩码中心与控制指令都合理后，再停掉测试
进程并改用 `--esp32`。同一台树莓派不要同时启动两个追踪进程。

## 1. 启动 3090 视觉服务

在 Windows PowerShell 连接 3090：

```powershell
ssh -i F:/SSHKeys/sjtu_ed25519 sjtu@192.168.55.33
```

在 3090 上执行：

```bash
cd ~/track
chmod +x pursuit_service.sh
./pursuit_service.sh restart --device cuda --no-depth
./pursuit_service.sh status
```

服务管理命令：

```bash
cd ~/track
./pursuit_service.sh start --device cuda --no-depth
./pursuit_service.sh stop
./pursuit_service.sh restart --device cuda --no-depth
./pursuit_service.sh status
./pursuit_service.sh logs
```

`logs` 会持续显示最近一次服务日志，按 `Ctrl+C` 只退出观察，不会关闭服务。

## 2. 打开观察网页

浏览器打开：

<http://192.168.55.33:8021/>

树莓派未启动时网页没有双摄实时画面；树莓派开始上传后，网页会显示左右画面、
追踪框、左右掩码目标点、感知帧率、相机/上传帧率、深度和车辆状态。网页中的
“重新识别”会在最新画面上重新调用 Qwen 并把结果发回树莓派状态机。

健康检查：

```text
http://192.168.55.33:8021/health
```

## 3. 启动树莓派（先测试）

连接树莓派，SSH 提示时输入密码：

```powershell
ssh dachuang@172.18.14.74
```

在树莓派执行无电机测试：

```bash
cd ~/dachuang/track
PYTHONUNBUFFERED=1 ~/dachuang/.venv/bin/python pursuit/run_visual_pursuit.py \
  --instruction "追踪那个蓝色瓶子" \
  --server-http http://192.168.55.33:8021 \
  --server-ws ws://192.168.55.33:8021/pursuit_ws \
  --dry-run
```

按 `Ctrl+C` 正常结束，并完整写入本次日志。

## 4. 启动实车

先结束 dry-run，架起车轮或清空车辆前方，确认网页双路目标框正确，再执行：

```bash
cd ~/dachuang/track
PYTHONUNBUFFERED=1 ~/dachuang/.venv/bin/python pursuit/run_visual_pursuit.py \
  --instruction "追踪那个蓝色瓶子" \
  --server-http http://192.168.55.33:8021 \
  --server-ws ws://192.168.55.33:8021/pursuit_ws \
  --esp32
```

默认启用 N10 雷达。仅在明确没有雷达时添加 `--no-lidar`；排障时可以添加
`--left-camera N --right-camera N` 或 `--swap-cameras`。默认每次运行都会独立
归档，不需要再传 `--log-csv`。

## 5. 查看树莓派任务日志

每次任务目录：

```text
~/dachuang/track/logs/pursuit/YYYYMMDD_HHMMSS_<dry_run|esp32>_<目标>/
```

目录内文件：

- `run.json`：指令、模式、启动参数、起止时间、最终状态及服务器地址；
- `console.log`：摄像头、Qwen、追踪、网络、BLE、控制状态的完整终端输出；
- `telemetry.csv`：每秒的双路 bbox、追踪状态、来源、感知年龄/帧率、相机与
  上传帧率、深度、雷达距离及运动命令。

查看最近一次运行：

```bash
cd ~/dachuang/track
cat logs/pursuit/LATEST.txt
latest=$(cat logs/pursuit/LATEST.txt)
tail -n 100 -F "$latest/console.log"
```

按 `Ctrl+C` 退出日志观察。生成最近一次运行的诊断摘要：

```bash
cd ~/dachuang/track
~/dachuang/.venv/bin/python pursuit/summarize_run.py
```

分析指定的旧任务：

```bash
~/dachuang/.venv/bin/python pursuit/summarize_run.py \
  logs/pursuit/20260820_160000_esp32_追踪那个蓝色瓶子
```

## 6. 查看 3090 服务日志

```bash
cd ~/track
./pursuit_service.sh logs
```

归档位于 `~/track/logs/server/YYYYMMDD_HHMMSS/server.log`，最近一次目录写在
`~/track/logs/server/LATEST.txt`。排查一次实车任务时，应同时保留树莓派任务
目录和同时间段的 3090 服务目录。

## 7. 后续反馈时应提供

出现错框、近距离丢失、转向过快或停车异常后，记录大致发生时间，并提供：

1. 树莓派最近一次完整任务目录；
2. 3090 同时间段的服务目录；
3. 网页截图或录屏（若有）；
4. 当时目标描述、光照、目标距离，以及车辆是否实际运动。

不要只截取 CSV 的几行；完整目录能把 Qwen 初始框、YOLO/模板追踪来源、帧率、
网络延迟和控制指令按时间对齐，后续才能复现原因。
