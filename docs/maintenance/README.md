# 维护与发布

维护工具位于 `tools/maintenance/`。修改后先执行离线验证，再按设备部署清单发布。

- `python tools/project.py check` 检查源码；场景地图及来源清单已配置时也核对相应数据。
- `python -m tools.maintenance.public_hygiene --staged` 检查Git暂存内容，报告文件路径和问题类别。
- `assemble_workspace.py` 等维护工具使用前，需按[配置说明](../configuration.md)配置设备连接。
- 不要整目录覆盖设备。按明确文件清单发布，先确认任务空闲、备份，并保留设备学习状态、地图、标定和私密配置。
- 源码更新不会自动在运行中的服务生效；发布后需核对版本并重载对应服务。

协作和提交要求见[CONTRIBUTING](../../CONTRIBUTING.md)。
