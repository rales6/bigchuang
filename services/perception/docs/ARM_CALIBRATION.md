# 机械臂姿态标定与启用门槛

任务网关只接受八个经过实车验证的姿态名，不接受自然语言模型给出的关节值。标定目标不是
“让动作看起来能运行”，而是为当前这台小车建立可重复、不会碰撞车体和地面的脉宽范围。

## 标定前条件

1. 架空车轮，清空机械臂活动区域，准备可立即断电的急停方式；由现场人员观察机械限位。
2. 确认小车唯一编号，标定文件不能跨机器人复用。
3. 核对 ESP32 固件和舵机编号仍为 `0..5`；当前链路支持 Vehicle Link V2
   `MSG_SET_ARM_JOINTS/MSG_ARM_STOP`，旧链路支持 `#000..#005` 与 `#255PDST!`。
4. 复制 `arm_profile.template.json`，整个标定期间保持 `calibrated=false`。

## 必须实测的数据

- 六个关节的真实安全最小/最大脉宽；软件的 400–2600 μs 只是外围保守界限，不能替代机械限位。
- `stow`、`pregrasp`、`grasp`、`lift`、`carry`、`preplace`、`place`、`open` 八个姿态。
- 每个姿态的完成时间，以及任意相邻姿态之间不会撞击车体、地面、相机或载荷。
- 相机中“目标居中且可达”的区域与夹爪工作区关系。
- 夹爪打开、闭合以及夹住典型垃圾/水瓶时的图像证据。

只有现场逐项通过后，才能把四个 `safety_checks` 改为 `true`，填写带时区的 ISO-8601
`calibrated_at`，最后设置 `calibrated=true`。

## 离线校验

校验命令不会打开串口、BLE、底盘或机械臂：

```bash
python validate_arm_profile.py arm_profile.calibrated.json \
  --robot-id <当前小车唯一编号>
```

必须看到：

```json
{"ready_for_mapper": true, "hardware_opened": false}
```

模板文件、占位字段、越界脉宽、缺少姿态、未完成安全检查、无时区时间或机器人编号不匹配
都会返回非零退出码。

## 分阶段现场验收

1. `--drive-dry-run` 下启动地图服务，确认 `/api/arm/status` 的 profile 和 robot id 正确。
2. 保持车轮架空，只测试 `stow`，确认急停和 `/api/arm/stop` 有效。
3. 逐一测试八个姿态；任何异常立即停机并恢复 `calibrated=false`。
4. 启动 3090 `/verify`，采集“未夹住/已夹住、未投放/已投放”的正负样本；低置信度必须阻断。
5. 最后才允许任务网关硬件模式，并逐次调用 `/execute/tick`，禁止 `auto_run`。

树莓派或 3090 失联、摄像头帧过期、雷达前方净空不足、底盘未停止、目标不可达、模型
置信度不足或任一姿态故障时，任务必须停在当前步骤，不能用超时当作成功。
