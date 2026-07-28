# 树莓派激光雷达建图与 ESP32 联调

## 1. 实现范围

树莓派侧代码位于 `raspberry_pi/`，职责分为三层：

- `esp32/client.py`：Vehicle Link V2 串口请求、CRC、应答校验、心跳和速度命令续期；
- `lidar/`：品牌无关的 `LaserScan` 接口和镭神 N10 原生串口驱动；
- `mapping/`：二维 ICP 扫描匹配、位姿累计、占据栅格和地图导出；
- `mapping_app.py`：把通信、雷达和建图模块组合成可直接运行的程序；
- `autonomous_mapping.py`：复用 car1.0 已验证 BLE 控制链路的自主前沿探索；
- `mapping_test.py`：不连接 ESP32 的前 180°手推建图测试；
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
ls -l /dev/ttyACM0
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

## 4. 先测试树莓派—ESP32 BLE 通信

先让车轮悬空、机械臂周围无障碍物。默认测试只发送心跳和查询状态，不会驱动车轮：

```bash
python3 -m raspberry_pi.communication_test \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --count 10
```

正常时会显示每次应答延迟、ESP32 uptime、状态位和关节位置。当前旧底板没有电池及
轮速回传，所以 `battery=0`、轮速为零不代表真实电压或真实静止。

确认基础链路后可测试 LED 和蜂鸣器：

```bash
python3 -m raspberry_pi.communication_test --link ble --io-test
```

只有车轮已经架空时才运行短时低速运动测试：

```bash
python3 -m raspberry_pi.communication_test --link ble --motion-test
```

该测试以 100 mm/s 请求运行 0.5 秒，随后发送 `STOP`，程序退出时还会发送全车
`CANCEL`。即便树莓派异常退出，ESP32 的 600 ms 命令 TTL 和 800 ms 失联保护也会停车。

## 5. 建图

让雷达水平安装，线缆不遮挡扫描平面。雷达坐标约定为 x 向车头、y 向车左、逆时针为
正。N10 厂商角度在俯视图中顺时针增加，驱动已经默认反向转换；只需要用
`--angle-offset` 修正雷达零度与车头之间的安装偏角。

先单独验证 N10。程序会检查帧头、58 字节长度和累加和，并显示点数、转速与距离范围：

```bash
python3 -m raspberry_pi.lidar_test --port /dev/ttyACM0 --count 5
```

默认会发送厂商定义的 188 字节电机启动命令，并在退出时发送停止命令。如果雷达由其他
控制器管理电机，可增加 `--no-motor-control`。

### 独立建图测试（推荐先运行）

不连接 ESP32、只验证雷达和建图算法时运行：

```bash
python3 -m raspberry_pi.mapping_test \
  --port /dev/ttyACM0 \
  --count 200 \
  --output maps/test_room
```

该测试现在默认只保留车头方向 180°，受车身影响的后半圈不会进入 ICP 和地图。
如果实际可用范围更小，可以进一步缩到 160°：

```bash
python3 -m raspberry_pi.mapping_test \
  --port /dev/ttyACM0 \
  --field-of-view 160 \
  --view-center 0 \
  --count 200 \
  --output maps/test_room
```

`--view-center 0` 表示有效区域中心朝向车头，因此保留约 `-90°..+90°`。如果雷达
安装方向有偏差，可先用 `--angle-offset` 校正雷达零度，或调整 `--view-center`。
被车身遮挡的角度必须过滤，否则车身反射会进入 ICP，并在地图中形成随机器人移动的
假障碍。

程序启动后可以缓慢、平稳地手推小车。不要快速转弯或搬起小车，否则相邻扫描重叠
不足，ICP 会拒绝该圈数据。也可以让雷达保持静止，先验证地图文件是否能正常生成。

程序运行 200 圈后自动退出；也可以按 `Ctrl+C` 提前保存。输出包括：

- `maps/test_room.pgm`：占据栅格地图；
- `maps/test_room.yaml`：ROS 兼容地图参数；
- `maps/test_room_trajectory.csv`：估计轨迹。

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `/dev/ttyACM0` | 雷达串口 |
| `--count` | `200` | 读取的完整扫描圈数 |
| `--map-size` | `20` | 正方形地图边长，米 |
| `--resolution` | `0.05` | 地图分辨率，米/格 |
| `--min-distance` | `0.12` | 最小有效距离，米 |
| `--max-distance` | `8.0` | 最大有效距离，米 |
| `--angle-offset` | `0` | 雷达零度相对车头的安装偏角 |
| `--field-of-view` | `180` | 保留的有效视场角，后半圈默认丢弃 |
| `--view-center` | `0` | 有效视场中心角；`0` 表示车头方向 |
| `--save-every` | `50` | 自动保存间隔，扫描圈数 |

如需更小、更快的桌面测试：

```bash
python3 -m raspberry_pi.mapping_test \
  --port /dev/ttyACM0 \
  --count 50 \
  --map-size 10 \
  --output maps/desk_test
```

该独立测试没有连接 ESP32，也不会控制电机、机械臂、LED 或蜂鸣器。

### BLE 自主覆盖建图

自主脚本不修改 car1.0 已验证的 BLE 实现，只调用同一个
`Esp32Client.set_twist()`。工作流程是：BLE 无动作心跳、在起点停车稳定建图、寻找
可达前沿、移动时只定位、到达下一个观察位置后停车并进行多圈融合。默认每平移
0.40 m 或转向累计达到 30° 安排一次停车观测。定位始终使用完整 360° 当前帧、完整
上一帧、最近 8 帧局部子图和长期关键帧；只有停车融合后写占据栅格与避障时才裁成
前方 160°。前沿目标默认取所有可达白色/未知边界中直线距离最远的点，失败目标附近
会暂时排除，但不会连带隐藏同一边界上的其他可达点。

首次必须架空车轮，限制为一分钟：

```bash
python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --lidar-port /dev/ttyACM0 \
  --usable-fov 160 \
  --mapping-max-distance 6.0 \
  --wall-gap-max 0 \
  --speed 250 \
  --max-drive-speed 460 \
  --slow-speed 150 \
  --turn-speed 1500 \
  --max-runtime-min 1 \
  --output maps/ble_bench
```

确认雷达 0°、车轮方向、原地转向和停车均正确后，再在封闭场地运行：

```bash
python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --lidar-port /dev/ttyACM0 \
  --usable-fov 160 \
  --mapping-max-distance 6.0 \
  --wall-gap-max 0 \
  --stationary-map-distance 0.40 \
  --stationary-map-angle 30 \
  --stationary-settle-scans 2 \
  --stationary-fusion-scans 5 \
  --stationary-min-support 3 \
  --stationary-range-tolerance 0.10 \
  --speed 300 \
  --max-drive-speed 460 \
  --slow-speed 180 \
  --turn-speed 1500 \
  --stop-distance 0.40 \
  --slow-distance 0.75 \
  --clearance 0.30 \
  --map-size 20 \
  --max-runtime-min 15 \
  --save-every 0 \
  --output maps/ble_scene_01
```

### 实时查看实车算法地图

在自主建图命令中增加 `--live-map`，树莓派会启动一个轻量级监控网页：

```bash
python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --lidar-port /dev/ttyACM0 \
  --usable-fov 160 \
  --speed 250 \
  --max-drive-speed 460 \
  --slow-speed 150 \
  --turn-speed 1500 \
  --max-runtime-min 15 \
  --live-map \
  --live-map-port 8766 \
  --output maps/ble_scene_live
```

程序启动后会打印局域网访问地址，例如：

```text
实车算法地图监控已启动：http://192.168.1.50:8766
```

让电脑或手机与树莓派连接同一局域网，然后在浏览器中打开该地址。页面显示的数据直接
来自当前进程内的 `LidarSlam`：

- N10 实际扫描经过 ICP 后估计的小车位姿；
- 当前占据栅格、自由空间和未知区域；
- 红色估计轨迹、绿色起点和蓝色当前位置；
- 扫描接受率、地图融合帧数、RMSE、匹配点数和内点率；
- 自主控制状态、下发速度、地图门控原因和定位拒绝信息。

网页不读取仿真器真值，也不会参与底盘控制。地图默认以 2 Hz 在后台线程中编码，
不会堵塞雷达、蓝牙和运动控制循环，也不会每帧写入磁盘；如果树莓派来不及渲染，
只丢弃旧的网页画面并保留最新一帧。状态指标仍随每个扫描帧更新。可通过
`--live-map-refresh-hz 1.0` 降低树莓派负载，允许范围为 `0.2..10.0` Hz。
`--live-map-bind 0.0.0.0` 是默认值，允许局域网访问；若只允许树莓派本机查看，可改为
`--live-map-bind 127.0.0.1`。如果网页无法打开，确认两台设备在同一网络，并允许 TCP
端口 `8766` 通过本机防火墙。

如果同名设备较多，可增加 `--ble-address AA:BB:CC:DD:EE:FF`。`--clearance`
至少应覆盖车体中心到最外缘的距离，并留出定位误差。运动命令默认只有 450 ms
有效，脚本不会启动后台无限续期；雷达停止出帧或程序卡住后，ESP32 TTL 会停车。
超过 `--mapping-max-distance` 的回波不会生成黑色远墙，只把该方向清为空闲到限距，
更远区域保持未探索。默认 `--wall-gap-max 0`，不会凭几何形状补墙；只有明确确认
环境墙体连续且雷达采样造成固定小缺口时，才应手动设置一个很小的非零值。弱单次
回波和小障碍点仍会在输出前删除；该清理只影响图像，不篡改定位与规划使用的概率
栅格。

自主建图已经不再把运动中的单圈雷达直接写入地图。进入
`state=stationary_mapping` 后先发送零速度，默认丢弃 2 圈用于等待底盘和雷达支架
稳定，再收集 5 圈。每圈先做单帧邻域离群点过滤，然后按 1° 方向分箱；每个方向
对距离取中值，只保留至少 3 圈支持且距离差不超过 0.10 m 的回波。最终只把这一份
融合后的静止关键帧写入长期地图。移动和转向期间的扫描仍用于定位、避障和车头方向
估计，但地图状态显示为 `moving_localization_only`，不会增加黑墙或清除旧边界。

融合结果写入栅格时会使用与 3/5 圈一致性相匹配的占据证据权重，因此一个可靠停车
关键帧就能越过黑墙显示阈值；端点在栅格中扩展一格，避免远墙相邻激光束间距较大而
被误判成零散小点。占据地图也不是只增不减：连续五个不同停车融合关键帧的自由射线
与旧墙冲突才会撤销旧墙，同一个融合关键帧中的多条射线只计一次；同一位置和朝向
保留四个独立融合关键帧，新观测每次最多淘汰一个最旧
证据。15 分钟运行会保留最多 1800 帧，不再因原来的 320 帧上限而把早期房间整片
删掉。自主模式默认关闭运行中的磁盘快照，避免树莓派渲染 PNG 时阻塞控制与 BLE
续期；最终停车保存时会把保留的关键帧重放到全新网格；
蓝牙短暂重连期间不再执行耗时重建。因此反复观测可以逐渐修正偏斜墙线，但单帧
新数据不能清空旧地图。

探索器会锁定当前最远前沿分支，不再按固定帧数切换目标。只有到达目标、当前路径
被障碍封死或持续移动却没有增加覆盖面积时，才将该分支暂时列入黑名单，并沿已知
空闲区返回去选择另一分支。车头角度从启动方向零点开始，使用完整360°相邻雷达
散点云的ICP旋转量累计；位置变化不用于推断车头方向。雷达证据接近地图边缘时，
栅格会保留原世界坐标并自动向相应方向扩展。矩形房间方向假设默认关闭；只有确认
场地确实由水平/垂直墙组成时才使用 `--manhattan` 显式启用。

导出图像只显示实际概率证据：可靠占据为黑色、射线确认可通行为白色、未观测区域
保持灰色。闭合轮廓内部不会再凭几何形状强行填黑，因为那不是雷达测量，容易制造
大面积假障碍。

四轮补偿值由树莓派随运动命令下发。左后轮容易空转时可先使用
`--front-left-gain 1.15 --rear-left-gain 0.90`，两个右轮保持 `1.00`；
所有倍率允许范围为 `0.50..1.50`。

`--speed` 是正常巡航速度；`--max-drive-speed` 是检测到静摩擦堵转或执行脱困动作时
允许逐级提升的安全上限。此前两者共用同一个上限，使用 `--speed 300` 时即使检测到
堵转也无法继续加力，并且 450 mm/s 的倒退脱困指令会被错误压到 300 mm/s。

前沿探索覆盖的是当前地图中“可通行、可观察”的未知区域，无法保证玻璃、台阶、
悬空障碍、窄门或动态场景的完整覆盖。二维雷达不能发现台阶和扫描平面之外的障碍，
实车运行必须有人监护并保留物理急停。

按 `Ctrl+C` 后程序先请求全车取消，再保存：

- `room_01.pgm`：占据栅格图像；
- `room_01.yaml`：ROS 地图元数据；
- `room_01_trajectory.csv`：每个有效扫描的时间戳和二维位姿。

自主模式默认 `--save-every 0`，只在正常结束或按 `Ctrl+C` 后保存，避免运行中出现
十几秒的渲染停顿。如确实需要断电保护，可显式设置较大的间隔，并先确认树莓派保存
耗时不会影响控制。

## 6. 更换雷达

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

## 7. 常见问题

- BLE 心跳超时：先单独运行 `communication_test --link ble`，确认广播名、地址和
  ESP32 固件与已经通过实车测试时一致。
- 无权限打开串口：把当前用户加入系统的串口用户组后重新登录，或按系统规则配置权限；
  不建议长期以 root 运行整个程序。
- N10 完全没有点云：确认打开的是雷达 USB 串口而不是 ESP32 串口，并确认 230400 波特率；
  若外部控制器已经启动雷达，可尝试 `--no-motor-control`。
- 地图左右镜像：当前驱动已按 N10 的顺时针角度自动转换；只有硬件实际输出方向相反时，
  才在建图程序中增加 `--counterclockwise`。
- 地图整体旋转：设置 `--angle-offset 90`、`180` 等实际安装偏角。
- 墙体重影：降低行驶速度，确认雷达固定牢靠；大面积空旷或长直走廊会让纯激光 ICP
  约束不足，需增加编码器里程计或使用带回环检测的 SLAM。
