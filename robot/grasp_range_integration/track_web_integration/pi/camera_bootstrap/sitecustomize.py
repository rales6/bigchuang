"""Only enabled for task-owned Pi subprocesses by camera_environment()."""
import os
if os.environ.get('FINAL_WORK_CAMERA_BROKER_URL'):
    try:
        from final_work.pi.camera_proxy import install_camera_proxy
        install_camera_proxy()
    except Exception as exc:
        # Python normally swallows sitecustomize exceptions. Do not continue
        # and reopen physical cameras owned by the recording daemon.
        raise SystemExit('共享摄像头适配初始化失败：'+str(exc)) from exc
