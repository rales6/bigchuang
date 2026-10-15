# 依赖按运行端分开

- 电脑离线检查：`python -m pip install -r requirements/dev.txt`。
- 树莓派建图/导航：`robot/requirements-pi.txt`，实际运行以设备虚拟环境为准。
- 视觉追踪：`services/track/pursuit/requirements.txt`。
- 3090 感知：`services/perception/pyproject.toml`、`services/perception/requirements/`。
- Android：Gradle wrapper，见 `apps/android/README.md`。

不要把 Windows 虚拟环境复制到树莓派，不要把 Torch/CUDA 或模型权重当普通源码提交。
