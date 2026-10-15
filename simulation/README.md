# 离线仿真

源码在 `car_sim/`，保留原 Python 包名，未收回旧 `output/` 和虚拟环境。

在总仓库根目录运行：

```bash
python tools/project.py test --suite simulation
```

该套件使用虚拟底盘/雷达，与真实硬件无连接。当前地图从 `maps/current/` 读取，临时输出在被忽略的 `.local/test-tmp/`；手动完整仿真输出应放 `runtime/simulation/`。

旧的单文件回放脚本可能带历史默认路径，先读其参数帮助，不把 `test_*` 或 `probe_*` 名称当作无硬件保证。默认 CI 不自动收集这些入口。
