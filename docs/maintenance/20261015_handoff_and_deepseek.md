# 2026-10-15 交接：到达判定、源码同步与 DeepSeek 配置

电脑总仓库：`D:\xingxianfeng_files\dachuang\robot-car`。`D:\xingxianfeng_files\dachaung` 是指向 `dachuang` 的 Windows Junction，不是另一个需要同步的仓库。Git 历史保留，未自动提交、推送或删除旧目录。

## 到达判定修改

最新参考任务 `20261015_201703` 最终成功，但从20:17:03到20:24:24约7分21秒，进行了11次目标观察。同一个室内沿墙木柜在多次画面中被识别为 `beige` / `light_brown`，直到最后输出 `yellow` 才被原规则接受；途中定位恢复也占用了时间，不能把全部耗时都归因于颜色。

修改入口：`services/track/final_work/pi/region_navigation.py`，开发镜像在 `robot/grasp_range_integration/track_web_integration/pi/region_navigation.py`。

- “附近”按目的地区域判定：距地图目标中心估计0.45–1.80m，或距规划观察站不超过0.60m；不再只接受距观察站0.45m内的点。仍需当前目标方向合理、至少一个相机确认指定目标，不能仅凭地图距离宣告成功。
- 首个观察方向的末段，在观察站0.60m内且目标区域已进入视野时可先停车拍照，不再为精确站位和精确最终朝向反复调整。仅作用于距最终观察站0.25m内的路径点；不跳过前方路线中间点，未确认后保留其余角度观察和换站。
- 仅对地图绑定、地图自身没有颜色标注的黄色柜子，接受置信度至少0.80的米色、浅棕色、浅木色等结构化主体颜色。不是从“黄色标签”等描述中挖颜色；白门、明确蓝/绿/红/白主体、玻璃后柜子仍排除。小物块颜色匹配不放宽。
- 搬运阶段的放置净空检查继续保留；本轮没有改电机控制、舵机偏置、夹取模型、玻璃边界或双目录像。

“停车拍照”不是任务成功；照片确认通过后才进入下一技能。上述米制距离来自估计地图位姿，不是视觉测距精度保证。

## 本地源码核对

按原始947项来源清单核对远端来源文件，另补充扫描当前功能目录内的新增源码。之前部署的导航、定位恢复、自适应转向、双目录像、靠近夹取、原地放下和技能编排均已保留在总仓库对应功能目录。

发现并补齐了一份旧本地副本：`services/grasp/pi/placement_extension.py`，现与设备 `/home/dachuang/grab_visual/placement_extension.py`、本地 `services/track/final_work/pi/placement_extension.py` 一致。未用树莓派中闲置的旧网页静态文件/开发镜像覆盖3090正在使用的新版本；本地三个新增/修订离线测试也未退回设备旧版。

`maps/current` 的15个当前地图文件哈希不变。照片参考清单的本地/远端JSON内容相同，只有序列化格式造成哈希差异，不覆盖。运行中学习到的速度经验另存 `.local/handoff/pi_motion_experience_20261015.json`，未把机器状态当源码、未导入旧录像和实验地图。

审计明细在本机 `.local/handoff/source_audit_before.json`、`source_tree_audit.json` 和 `source_sync_final.json`；部署备份/逐文件哈希在 `.local/deployments/`。这些是忽略的维护证据，不是运行依赖。

最终636项远端/本地对应关系中：615项哈希相同，3项本地测试较新，1项运行经验另存，1项JSON仅格式不同，3项设备闲置开发镜像较旧，3项Pi闲置网页静态文件由3090正式版本取代。无未归类差异、无缺失源码；本地36个正式/开发镜像Python文件全部一致。

## DeepSeek 密钥：由原成员自行清空

用户明确选择“只列出位置，我自行清空”。本次没有清空、替换、撤销或在文档中记录任何真实密钥；也未删除调用代码。以下为本轮只读检查结果，不是未来状态保证。

| 位置 | 内容及本轮结果 | 清理入口 |
| --- | --- | --- |
| 树莓派 `/home/dachuang/dachuang/car1.0/secrets/deepseek_vision.json` | `api_key` 当前有值；现用视觉复核、规划和网页机器人技能共用 | 自行编辑此JSON中的 `api_key`。需停用调用时同时设 `enabled=false`；不要把此文件复制进Git |
| 3090 `/home/sjtu/track/final_work/data/history.sqlite3` | `app_settings` 表，`key=api_credentials`，JSON的 `deepseek_api` / `deepseek_flash_api` 本轮均为空 | 网页设置选择对应供应商并清除密钥；不要删除整库，里面还有任务与历史 |
| 3090 `/home/sjtu/track/final_work/data/backups/history_before_cleanup_20261003_184116.sqlite3` | 同一设置字段，本轮均为空 | 历史数据库也应纳入交接审查；本轮没有删除备份 |
| 使用网页的每台电脑/手机浏览器 | 当前网页源站的 `localStorage['final_work.deepseek_flash_api_key']`，本轮未读取浏览器存储 | 网页设置→DeepSeek Flash→清除密钥。每个浏览器、用户配置及不同IP/域名源站分别处理 |
| 启动服务环境 | `FINAL_WORK_DEEPSEEK_API_KEY` 可为普通DeepSeek API提供密钥；本轮检查的活动网页/桥进程未发现DeepSeek环境变量 | 若以后配置过，检查实际启动脚本、`.env`、systemd `Environment` / `EnvironmentFile`，移除后重载对应服务 |
| Pi `/etc/systemd/system/robot-companion.service.d/relationship-map.conf` | `DEEPSEEK_VISION_CONFIG` 指向配置文件；不是另一个API密钥字段 | 核对所指文件。发布目录 `robot-companion-relationship-override.conf` 和本地 `robot/{scene_runtime,tools}/` 有对应模板/副本，完整路径在索引中 |
| 本地 `.env.example` | `DEEPSEEK_API_KEY` 为空的示例占位，不是已启用的运行配置 | 保留模板即可；该变量不等于网页服务的 `FINAL_WORK_DEEPSEEK_API_KEY` |

当前网页源站是 `http://192.168.55.33:8030`。如需自行在该页面的开发者工具控制台清除浏览器副本，可执行：

```javascript
localStorage.removeItem('final_work.deepseek_flash_api_key');
```

树莓派的配置路径还可能被 `DEEPSEEK_VISION_CONFIG`（视觉定位）或 `DEEPSEEK_CONFIG`（`plan_phase1_deepseek.sh`）覆盖。网页 `skill_planning.py` 则固定读取其 `car1_root/secrets/deepseek_vision.json`。若后续换路径，不能只清理默认文件。密钥清空后，使用该密钥的规划/视觉复核将不可用，新成员应使用自己的配置；本地Qwen识别代码不会因此被删除。

删除本地配置不等于撤销账户密钥。交接前还应在 DeepSeek 供应商控制台撤销旧密钥/新建团队自己的密钥，以覆盖未知备份、旧进程内存或浏览器副本。源码备份、Git历史和个人浏览器不在本次配置清空范围内；本次也未声称已完成账户级撤销。

## API 调用代码与空模板位置

以下均为总仓库相对路径。它们是读取配置/发起请求的代码，不能因为包含“DeepSeek”就整文件删除。

| 功能 | 文件 |
| --- | --- |
| 雷达候选的视觉复核 | `robot/scene_runtime/deepseek_visual_localization.py` |
| 第一阶段自然语言规划与入口 | `robot/grasp_range_integration/phase1/deepseek_planner.py`、`plan_phase1_deepseek.sh` |
| 目标颜色/视角恢复复核 | `robot/grasp_range_integration/phase1/colour_target_review.py`、`red_target_deepseek.py`、`probe_stereo_deepseek_pose.py`、`observation_recovery.py` |
| 网页技能规划，Pi保存密钥 | `services/track/final_work/pi/skill_planning.py`、`robot_bridge.py` |
| 网页供应商设置/调用 | `services/track/final_work/gpu/app.py`、`planner.py` |
| 浏览器本地密钥与直连请求 | `services/track/final_work/static/app.js` |
| 仅代理请求、不保存配置的本地桥 | `services/track/final_work/tools/deepseek_local_bridge.py` |
| 旧版入口（不是当前网页入口） | `services/track/gpu/app.py`、`services/track/gpu/planner.py`、`services/track/tools/deepseek_local_bridge.py` |
| 开发镜像 | `robot/grasp_range_integration/track_web_integration/{pi,gpu}/`；修改时与正式目录对应文件保持一致 |
| 空JSON模板 | `robot/grasp_range_integration/config/deepseek_phase1.example.json`、`robot/tools/deepseek_vision_config.example.json`、`maps/current/deepseek_vision_config.example.json` |

完整路径索引见 [deepseek_api_inventory.json](deepseek_api_inventory.json)：包含本地、Pi、3090所有扫描到的DeepSeek引用文件路径，以及配置字段名/是否有值，**不包含密钥内容**。旧 `car1.0` 中两个路径别名下的 `tests/test_deepseek_visual_localization.py` 也被“密钥形状”检测命中，它们是测试位置，不能等同于现用配置。索引中的日志/结果引用不是独立密钥存储位置。

可重复运行的只读工具：

```powershell
# 在总仓库，SSH密码只由当前进程环境提供；不要写进脚本或提交到Git。
.\.venv\Scripts\python.exe -m tools.maintenance.inventory_deepseek --remote
```

扫描覆盖总工作区、两端项目/备份目录、systemd单位和已知网页设置数据库；跳过Git对象、虚拟环境、模型、录像、`.local`和大于3MiB的文本，不检查浏览器或供应商账户。索引列出排除项，不能当作任意格式/任意位置的全盘秘密扫描保证。

## 验证

本轮离线 smoke 69项、regression 252项，共321项通过；静态检查657个Python文件、15个当前地图文件通过。隐私测试验证扫描报告不包含配置密钥、数据库密钥或其他供应商密钥，数据库不被修改。部署只更新源码、重载服务，不下发导航、夹取、放置动作；现场效果仍需新一轮实机验收。

部署备份 `runtime_20261015_203900`，13项目标文件编译、上传及哈希验证通过；网页进程由522861重载为523416，Pi桥、隧道与网页健康检查通过。部署后任务为空闲，仅恢复被动相机预览；持物状态与密钥未重建、未修改。
