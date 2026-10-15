# 下一组同学：GitHub 交接入口

仓库：<https://github.com/rales6/bigchuang>；交接分支：`handoff/20261016`（2026-10-16）。保留原 `miller` 分支历史，不直接覆盖 `main`、`miller` 或其他成员分支。先使用此分支查看最新内容，是否合入主分支由团队审核决定。

```bash
git clone --branch handoff/20261016 https://github.com/rales6/bigchuang.git robot-car
cd robot-car
```

## 先读什么

1. 根 [README](../README.md)：功能目录、网页/桥服务启动与离线检查命令。
2. [功能交接索引](features.md)、[组件边界](architecture.md)：建图、标注、自然语言规划、区域导航、识别、靠近微调、模型夹取、搬运、地面放置、双目录像各自的位置。
3. [最新修复、同步和 DeepSeek 清理位置](maintenance/20261015_handoff_and_deepseek.md)：到达判定、当前地图、配置和完整API调用路径索引。
4. [依赖说明](../requirements/README.md)：按电脑、Pi、3090分别安装，不把整个项目当成一个服务启动。

## 提交内容与必须另行配置的内容

本分支包含 Android、ESP32 固件、Pi/3090 源码、当前语义/雷达地图及必要定位参考、夹取模型服务和训练代码、小型测试夹具、仿真、工具、文档和离线CI。不因上传改变小车现有运行版本，不自动启动任何硬件。

以下不上传：真实API密钥/SSH密码/硬件令牌、任务数据库、运行日志、双目视频、旧地图实验、虚拟环境、构建产物、大模型与机械臂模型权重。当前地图的原始字节由 `.gitattributes` 保留，避免Windows/Linux换行差异破坏发布哈希。

需要向原成员交接设备访问权限，并另行取得/核对：

| 项目 | 现有设备位置或配置入口 |
| --- | --- |
| Pi 地图任务根 | `/home/dachuang/dachuang/car1.0/`，发布目录 `ble_scene_17_relationship_20260919/` |
| Pi 网页桥与追踪服务 | `/home/dachuang/dachuang/track/` |
| Pi 机械臂服务 | `/home/dachuang/grab_visual/` |
| 3090 网页/规划/感知服务 | `/home/sjtu/track/`、`/home/sjtu/3090/` |
| 当前机械臂模型权重及示教元数据 | `/home/sjtu/grab_visual/arm_teach_artifacts/model.pt`、`model_state.json`；哈希来源见 `maintenance/source_manifest.json`。Git不提供 `model.pt` |
| Qwen/YOLO 模型 | 3090现有模型目录；以各组件配置为准，不能仅凭旧README猜路径 |
| DeepSeek 现用配置 | Pi `car1.0/secrets/deepseek_vision.json`；由团队填自己的密钥，不提交Git |
| 网页浏览器密钥 | 原成员在每个浏览器清除；新成员自行配置。位置见上述DeepSeek交接清单 |
| 实车自适应速度经验 | Pi 发布目录 `grasp_range_integration/phase1/runtime/{motion_experience,startup_drive_experience}.json`；保留设备现有文件，不用Git覆盖。新机器需要继承时单独迁移，不强行加入源码仓库 |

原设备IP、用户名及绝对路径只代表交接时布局，换网络或电脑后要更新。没有随仓库分发密码。旧密钥由原成员自行清空并在供应商账户撤销，本次上传不会代为清空/撤销。

## 首次离线核对

在Windows PowerShell中：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.\.venv\Scripts\python.exe tools/project.py check
.\.venv\Scripts\python.exe tools/project.py test --suite smoke
.\.venv\Scripts\python.exe tools/project.py test --suite regression
```

Linux/macOS 使用 `.venv/bin/python`。这些命令不连接设备、不驱动电机/机械臂；离线通过不等于实车验收。完整历史保留用于来源核对；不要用浅克隆缺少的历史提交误判代码不完整。

## 继续开发与发布

- `robot/raspberry_pi` 和 `services/track/raspberry_pi` 服务于不同运行根，不直接互相覆盖。
- 正式网页代码在 `services/track/final_work/`，兼容开发镜像在 `robot/grasp_range_integration/track_web_integration/`，修改后核对对应文件一致。
- 先在自己的分支修改、离线验证，再确认设备空闲、备份并按明确发布清单部署；不要把新目录整体覆盖到Pi。
- 不在演示前顺手修改已标定的相机左右顺序、底盘协议、玻璃边界、模型与舵机参数。需要实车动作时另行安排现场看护。
- `main` 中其他成员的功能不应因交接重组被删除；通过拉取请求审阅合并，禁止强制推送覆盖团队历史。
