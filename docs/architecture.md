# 组件边界与目录原则

本次做的是仓库整合，不是重写运动算法。冻结或经实机验证的执行器保留原源码与 Python 包名，外围增加统一文档、离线检查、来源记录和数据排除规则。

```text
robot-car/
├─ apps/android/                 手机指令与状态显示
├─ firmware/car/                 ESP32 执行器、协议、驱动
├─ hardware/cad/                 机械结构图
├─ robot/
│  ├─ raspberry_pi/              建图、雷达、导航执行基础
│  ├─ grasp_range_integration/    语义任务、观察视野、恢复与网页开发适配
│  ├─ tools/                     原建图/标注/定位工具（兼容原导入名）
│  └─ tests/                     原组件回归测试及小型必要夹具
├─ services/
│  ├─ perception/                3090 Qwen/YOLO 感知项目
│  ├─ track/                     设备上的视觉追踪运行根目录
│  │  ├─ pursuit/                接近、微调与目标找回
│  │  ├─ final_work/{pi,gpu}/     任务网页、技能编排、录像
│  │  └─ integrations/           共享示教区域协议
│  └─ grasp/{pi,gpu}/             双目机械臂客户端与模型服务
├─ simulation/car_sim/           离线仿真源码与网页
├─ maps/current/                 当前地图发布数据
├─ tools/maintenance/            总仓库检查、收回与归档工具
├─ tests/                       总仓库离线验收入口
├─ requirements/                分运行端的依赖说明
└─ docs/                        功能、数据、维护与交接文档
```

`robot/raspberry_pi` 与 `services/track/raspberry_pi` 来自不同部署根：前者是地图任务运行依赖，后者是追踪服务使用的驱动。保留两者及哈希，不在没有实车验证时合并同名文件。

`robot/grasp_range_integration/track_web_integration` 是原适配层的兼容开发路径；`services/track/final_work` 是实际运行路径镜像。同步来源明确记录；不要只改其中一份就声称已在服务生效。后续如统一部署，应先用发布清单消除重复，不依赖手动复制。

设备上的 `/home/robot/dachuang/car1.0`、`/home/robot/dachuang/track`、`/home/robot/grab_visual` 和 3090 路径没有改动。新的电脑目录不是已经部署好的设备根目录，不能整目录覆盖设备。真实启动、依赖以及相机顺序按组件原说明，配置通过本地环境文件传入。
