# 树莓派激光雷达建图与 ESP32 联调

## 1. 实现范围

树莓派侧代码位于 `raspberry_pi/`，职责分为三层：

- `esp32/client.py`：Vehicle Link V2 串口请求、CRC、应答校验、心跳和速度命令续期；
- `lidar/`：品牌无关的 `LaserScan` 接口和镭神 N10 原生串口驱动；
- `mapping/`：二维 ICP 扫描匹配、位姿累计、占据栅格和地图导出；
- `mapping_app.py`：把通信、雷达和建图模块组合成可直接运行的程序；
- `autonomous_mapping.py`：前 180° 过滤、前沿规划、避障和自主覆盖建图；
- `communication_test.py`：先做无运动联调，再按需测试灯、蜂鸣器和底盘。

树莓派和 ESP32 共用 `car/protocol/` 中的协议实现，没有复制第二份 CRC 或消息编号。
ESP32 仍只负责确定性的执行和失联停车；地图、位姿和雷达数据不发送到 ESP32。

当前底板的旧 UART 协议不提供可信轮速反馈，因此本版本用相邻激光帧 ICP 估计位姿。
它适合室内原型和小范围建图，但没有回环检测，长距离闭环路线会累计漂移。若以后底盘
能回传编码器里程计，可把里程计增量作为 ICP 初值；正式大场景项目建议接入 ROS 2 的
SLAM Toolbox 或 Cartographer，本项目的雷达接口和 ESP32 控制协议无需改变。

## 2. 接线

树莓派与 ESP32 使用专用 3.3 V TTL UART：

| 树莓派 | ESP32 | 说明 |
|---|---|---|
| GPIO14 / TXD（物理脚 8） | GPIO21 / UART1 RX | 树莓派发、ESP32 收 |
| GPIO15 / RXD（物理脚 10） | GPIO22 / UART1 TX | ESP32 发、树莓派收 |
| GND（如物理脚 6） | GND | 必须共地 |

不要连接两块板的 5 V 信号脚，也不要让树莓派串口同时作为 Linux 登录控制台。N10 推荐
使用附送的 USB 转串口线单独连接树莓派。厂商规格书标注雷达供电为 5 VDC（允许范围
4.75-5.25 V）、通信为 230400 bps 标准异步串口；不要用树莓派普通 GPIO 给雷达供电。

在 Raspberry Pi OS 中用 `raspi-config` 启用串口硬件并关闭串口登录 Shell，重启后确认：

```bash
ls -l /dev/serial0 /dev/ttyUSB0
```

如果 USB 雷达被识别成其他设备名，以实际名称为准。多个 USB 串口同时存在时，建议在
`/dev/serial/by-id/` 下选择稳定设备路径。

## 3. 安装

把整个项目复制到树莓派。不要只复制 `raspberry_pi/`，因为它会复用 `car/protocol/`
中的 Vehicle Link V2 定义。在项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt
```

N10 串口版固定按 230400、8 数据位、无校验、1 停止位打开。这里实现的是 N10 的
58 字节点云协议，不适用于 N10P；N10P 是另一种帧长和波特率，不能直接混用。

## 4. 先测试树莓派—ESP32 通信

先让车轮悬空、机械臂周围无障碍物。默认测试只发送心跳和查询状态，不会驱动车轮：

```bash
python3 -m raspberry_pi.communication_test --port /dev/serial0 --count 10
```

正常时会显示每次应答延迟、ESP32 uptime、状态位和关节位置。当前旧底板没有电池及
轮速回传，所以 `battery=0`、轮速为零不代表真实电压或真实静止。

确认基础链路后可测试 LED 和蜂鸣器：

```bash
python3 -m raspberry_pi.communication_test --port /dev/serial0 --io-test
```

只有车轮已经架空时才运行短时低速运动测试：

```bash
python3 -m raspberry_pi.communication_test --port /dev/serial0 --motion-test
```

该测试以 100 mm/s 请求运行 0.5 秒，随后发送 `STOP`，程序退出时还会发送全车
`CANCEL`。即便树莓派异常退出，ESP32 的 600 ms 命令 TTL 和 800 ms 失联保护也会停车。

## 5. 建图

让雷达水平安装，线缆不遮挡扫描平面。雷达坐标约定为 x 向车头、y 向车左、逆时针为
正。N10 厂商角度在俯视图中顺时针增加，驱动已经默认反向转换；只需要用
`--angle-offset` 修正雷达零度与车头之间的安装偏角。

先单独验证 N10。程序会检查帧头、58 字节长度和累加和，并显示点数、转速与距离范围：

```bash
python3 -m raspberry_pi.lidar_test --port /dev/ttyUSB0 --count 5
```

默认会发送厂商定义的 188 字节电机启动命令，并在退出时发送停止命令。如果雷达由其他
控制器管理电机，可增加 `--no-motor-control`。

```bash
python3 -m raspberry_pi.mapping_app \
  --esp-port /dev/serial0 \
  --lidar-port /dev/ttyUSB0 \
  --output maps/room_01
```

默认地图为 20 m × 20 m、分辨率 0.05 m/格、起点在地图中心。可例如改成 30 m：

```bash
python3 -m raspberry_pi.mapping_app --map-size 30 --resolution 0.05 \
  --output maps/large_room
```

`mapping_app` 不会自主控制车辆，可以缓慢手推小车采图。它现在默认只保留车头方向
±90°，因此受车身遮挡的后半圈不会进入 ICP 或地图。若实际可用范围不是严格的 180°，
可使用 `--usable-fov 160`；若可用扇区中心相对车头偏左 5°，使用
`--front-center 5`。`--angle-offset` 用于先把雷达原始零度校准到车头方向，
`--front-center` 则是在校准后的车体坐标中选择可用扇区，两者不要混淆。

## 6. 前 180° 自主覆盖建图

自主脚本采用保守的“前沿探索”：

1. 原地慢速旋转一圈，让前向雷达观察四周；
2. 在地图中寻找已知空闲区与未知区交界的前沿；
3. 在经过障碍膨胀的空闲区内规划到最近可达前沿；
4. 对准路径后低速前进，每 8 个有效扫描重新规划；
5. 前方小于慢行距离时减速，小于停车距离时只原地转向，绝不倒车；
6. 暂时找不到前沿时原地复查一整圈；确认仍无可达未知区后停车并保存；
7. 连续定位失败、运行超时或收到 Ctrl+C 时立即停车并保存。

首次测试必须架空车轮，确认雷达正前方对应 0°，并把速度设得很低：

```bash
python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link uart \
  --esp-port /dev/serial0 \
  --lidar-port /dev/ttyUSB0 \
  --speed 50 \
  --slow-speed 30 \
  --max-runtime-min 1 \
  --output maps/bench_test
```

确认转向方向、定位方向和停车均正确后，再放到平整、封闭且有人可随时断电的场地：

```bash
python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link auto \
  --usable-fov 180 \
  --front-center 0 \
  --speed 100 \
  --stop-distance 0.40 \
  --slow-distance 0.75 \
  --clearance 0.30 \
  --map-size 20 \
  --max-runtime-min 15 \
  --output maps/scene_01
```

`--clearance` 至少应是车体中心到最外缘的距离，并额外留出定位误差。默认运动命令
只有 450 ms 有效，脚本不会启动后台无限续期；如果雷达停止出帧或主进程卡住，
ESP32 会在 TTL 到期后停车。串口只能由一个进程持有，运行自主脚本时不要同时打开
控制台或 `mapping_app`。

前沿探索可以覆盖“当前地图中可通行且能被雷达逐步观察到”的区域，但不能数学保证
任何场景都完整覆盖：窄于车身安全宽度的门、玻璃/镜面、台阶、悬空障碍、动态行人
以及纯 ICP 的累计漂移都可能导致遗漏或危险。当前二维雷达不能识别台阶和桌面高度
之外的障碍，真实自主运行仍应配置急停和人工监护。大场景建议增加编码器/IMU，并改用
带回环检测的 ROS 2 SLAM Toolbox 或 Cartographer。

按 `Ctrl+C` 后程序先请求全车取消，再保存：

- `room_01.pgm`：占据栅格图像；
- `room_01.yaml`：ROS 地图元数据；
- `room_01_trajectory.csv`：每个有效扫描的时间戳和二维位姿。

程序也会默认每 50 个有效扫描覆盖保存一次，避免意外断电丢失全部结果。

## 7. 更换雷达

新增驱动时实现 `raspberry_pi/lidar/base.py` 中的两个方法即可：

```python
class MyLidar:
    def scans(self):
        yield LaserScan(angles_rad, distances_m, timestamp_s)

    def close(self):
        pass
```

角度必须是弧度、距离必须是米，扫描点必须位于 x 向前/y 向左的车体坐标系。然后在
`mapping_app.py` 中把 `N10LidarDriver` 替换为新驱动，建图算法无需修改。

## 8. 常见问题

- 心跳全部超时：检查 TX/RX 是否交叉、是否共地、两端是否都是 230400 波特率，以及
  ESP32 是否已部署当前 `car/` 固件。
- 无权限打开串口：把当前用户加入系统的串口用户组后重新登录，或按系统规则配置权限；
  不建议长期以 root 运行整个程序。
- N10 完全没有点云：确认打开的是雷达 USB 串口而不是 ESP32 串口，并确认 230400 波特率；
  若外部控制器已经启动雷达，可尝试 `--no-motor-control`。
- 地图左右镜像：当前驱动已按 N10 的顺时针角度自动转换；只有硬件实际输出方向相反时，
  才在建图程序中增加 `--counterclockwise`。
- 地图整体旋转：设置 `--angle-offset 90`、`180` 等实际安装偏角。
- 后方车身轮廓仍进入地图：减小 `--usable-fov`，并确认 `--angle-offset` 已把 0°
  对准车头；例如先试 `--usable-fov 160`。
- 自主脚本一直原地转：通常是没有形成足够大的可达空闲区、ICP 连续失败，或安全
  膨胀半径大于通道宽度；先查看轨迹和地图，不要直接减小安全距离。
- 墙体重影：降低行驶速度，确认雷达固定牢靠；大面积空旷或长直走廊会让纯激光 ICP
  约束不足，需增加编码器里程计或使用带回环检测的 SLAM。
