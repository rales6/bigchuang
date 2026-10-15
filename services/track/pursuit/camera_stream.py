# -*- coding: utf-8 -*-
"""双 USB 摄像头自动探测与持续采集。

探测思路与 car1.0 的 ``map_camera_inspection.CameraCapture`` 一致：
优先使用 /dev/v4l/by-path、by-id 稳定节点（每个物理摄像头一个 index0
采集节点），解析出 /dev/videoN 整数编号后用 V4L2 打开；稳定节点不足
两路时回退到遍历 /dev/video0..15。只认能按目标分辨率/格式实际取到帧
的设备，避免选中视频采集卡、元数据节点等虚拟设备。
"""

from __future__ import annotations

import glob
import threading
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np

from pursuit.pursuit_config import CameraConfig


def frame_is_healthy(frame: np.ndarray | None) -> bool:
    """Reject frames decoded from a truncated MJPEG payload.

    Under USB load these cameras may return ``ok=True`` while OpenCV fills
    the undecodable lower image with neutral 128-gray.  Such a frame used to
    pass the FPS probe and then reach Qwen/YOLO.  Sparse sampling keeps this
    guard cheap enough to run in both capture threads.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        return False
    sample = frame[::8, ::8]
    neutral_fill = np.all((sample >= 126) & (sample <= 130), axis=2)
    return float(np.mean(neutral_fill)) < 0.35


def _stable_usb_nodes() -> list[str]:
    """返回每个物理 USB 摄像头一个稳定 udev 节点（去重后按路径排序）。"""
    nodes: list[str] = []
    resolved: set[str] = set()
    for pattern in (
        "/dev/v4l/by-path/*xhci*usb*-video-index0",
        "/dev/v4l/by-id/*-video-index0",
    ):
        for node in sorted(glob.glob(pattern)):
            try:
                target = str(Path(node).resolve())
            except OSError:
                target = node
            if target in resolved:
                continue
            resolved.add(target)
            nodes.append(node)
        if len(nodes) >= 2:
            return nodes
    return nodes


def _index_of(node: str) -> int:
    """稳定节点解析为 OpenCV 需要的整数编号。"""
    try:
        resolved = Path(node).resolve()
    except OSError:
        resolved = Path(node)
    name = resolved.name
    if name.startswith("video") and name[5:].isdigit():
        return int(name[5:])
    text = str(node)
    if text.startswith("/dev/video") and text[10:].isdigit():
        return int(text[10:])
    raise RuntimeError(f"无法解析摄像头编号: {node} -> {resolved}")


def probe_usb_cameras(config: CameraConfig) -> list[tuple[int, str]]:
    """探测两路可用的 USB 摄像头，返回 [(编号, 稳定路径), ...]。

    显式配置 ``left_device/right_device`` 时优先使用并跳过探测。
    """
    import cv2 as _cv2

    requested: list[tuple[int, str]] = []
    slots: list[int] = []
    for slot, index in enumerate((config.left_device, config.right_device)):
        if index >= 0:
            requested.append((index, f"/dev/video{index}"))
            slots.append(slot)
    if len(requested) == 2:
        return requested

    candidates = _stable_usb_nodes()
    if len(candidates) < 2:
        candidates = [f"/dev/video{i}" for i in range(16)]

    found: list[tuple[int, str]] = []
    failures: list[str] = []
    for node in candidates:
        try:
            index = _index_of(node)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        if any(index == used for used, _ in (requested + found)):
            continue
        cap = _cv2.VideoCapture(index, _cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                failures.append(f"/dev/video{index} 无法打开")
                continue
            cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*config.fourcc))
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            cap.set(_cv2.CAP_PROP_FPS, config.fps)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
            frame = None
            for _ in range(8):
                ok, candidate = cap.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
            if frame is None:
                failures.append(f"/dev/video{index} 无法取帧")
                continue
            found.append((index, node))
            print(f"[Camera probe] /dev/video{index} OK ({node}), "
                  f"frame={frame.shape[1]}x{frame.shape[0]}")
            if len(found) == 2:
                break
        finally:
            cap.release()

    if len(found) + len(requested) < 2:
        raise RuntimeError(
            f"只找到 {len(found) + len(requested)} 路可用 USB 摄像头: "
            f"{'; '.join(failures) or '无候选节点'}"
        )
    # 按左/右槽位组装：显式指定的固定在各自槽位，探测结果补入空槽。
    devices: list[tuple[int, str] | None] = [None, None]
    for slot, device in zip(slots, requested):
        devices[slot] = device
    for device in found:
        devices[devices.index(None)] = device
    assert devices[0] is not None and devices[1] is not None
    return [(devices[0][0], devices[0][1]), (devices[1][0], devices[1][1])]


def _measure_tier(devices: list[tuple[int, str]], width: int, height: int,
                  fps: int, fourcc: str,
                  buffer_size: int = 2) -> tuple[int, int] | None:
    """两路同时开测该档位 1s 内的实际帧率（USB 总线带宽共享，
    单路测不代表真实瓶颈）。返回 (左fps, 右fps)，打不开返回 None。"""
    import cv2 as _cv2

    caps: list = []
    try:
        for index, _ in devices:
            cap = _cv2.VideoCapture(index, _cv2.CAP_V4L2)
            cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*fourcc))
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(_cv2.CAP_PROP_FPS, fps)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            if not cap.isOpened():
                return None
            caps.append(cap)
        # 预热丢弃（模式切换后首帧慢）
        deadline = monotonic() + 0.4
        while monotonic() < deadline:
            for cap in caps:
                cap.read()
        counts = [0, 0]
        deadline = monotonic() + 1.0
        while monotonic() < deadline:
            for i, cap in enumerate(caps):
                ok, frame = cap.read()
                if ok and frame_is_healthy(frame):
                    counts[i] += 1
        return counts[0], counts[1]  # type: ignore[return-value]
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


def probe_camera_modes(config: CameraConfig,
                       devices: list[tuple[int, str]]) -> tuple[int, int, int]:
    """自动探测最佳采集模式，返回 (width, height, fps)。

    档位按优先级逐档实测（两路同时测，min(左,右) ≥ 阈值即达标）：
      1280x720@30（≥24fps）→ 1280x720@15（≥13）→ 960x540@30（≥24）
      → 640x480@30（≥24）。
    全部不达标取最低档继续运行（不崩溃，日志告警）。
    显式配置（指定了摄像头编号，或 width/height/fps 非默认，
    或 auto_probe=False）时跳过探测直接用配置。
    """
    explicit = (not config.auto_probe
                or config.left_device >= 0 or config.right_device >= 0
                or config.width != 1280 or config.height != 720
                or config.fps != 30)
    if explicit:
        return config.width, config.height, config.fps
    tiers = [(1280, 720, 30, 24.0), (1280, 720, 15, 13.0),
             (960, 540, 30, 24.0), (640, 480, 30, 24.0)]
    for width, height, fps, threshold in tiers:
        try:
            measured = _measure_tier(devices, width, height, fps,
                                     config.fourcc, config.buffer_size)
        except Exception as exc:
            measured = None
            print(f"[camera probe] {width}x{height}@{fps} 测量失败: {exc}")
        ok = measured is not None and min(measured) >= threshold
        print(f"[camera probe] {width}x{height}@{fps}: "
              f"{measured if measured else '无法打开'}"
              f" {'✓ 采用' if ok else ''}")
        if ok:
            return width, height, fps
    print("[camera probe] 全部档位不达标，取 640x480@30 继续（可能掉帧）")
    return 640, 480, 30


class CameraReader(threading.Thread):
    """后台线程持续取最新帧，只保留最新一帧供拼接。"""

    def __init__(self, name: str, index: int, config: CameraConfig,
                 notify: threading.Event | None = None) -> None:
        super().__init__(daemon=True, name=f"camera-{name}")
        self.name = name
        self.index = index
        self.config = config
        # 有新帧时通知拼接工作线程（线程安全：set 任意线程可调）
        self.notify = notify
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        self._frame: np.ndarray | None = None
        self._frames = 0
        self._read_ms_sum = 0.0
        self._last_error = ""
        self._seq = 0

    def run(self) -> None:
        import cv2 as _cv2
        try:
            cap = _cv2.VideoCapture(self.index, _cv2.CAP_V4L2)
            cap.set(_cv2.CAP_PROP_FOURCC,
                    _cv2.VideoWriter_fourcc(*self.config.fourcc))
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            # 2026-08-17 对齐 linux_camera/opencv_camera.py：V4L2 摄像头
            # 不设 FPS 会静默回退到默认低帧率。
            cap.set(_cv2.CAP_PROP_FPS, self.config.fps)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            if not cap.isOpened():
                self._last_error = f"/dev/video{self.index} 打开失败"
                return
            eff_fps = cap.get(_cv2.CAP_PROP_FPS)
            eff_w = int(round(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)))
            eff_h = int(round(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"[Camera {self.name}] /dev/video{self.index} requested "
                  f"{self.config.width}x{self.config.height}@{self.config.fps}; "
                  f"effective {eff_w}x{eff_h}@{eff_fps:.1f}")
            while not self.stop_event.is_set():
                started = monotonic()
                ok, frame = cap.read()
                read_ms = (monotonic() - started) * 1000.0
                if ok and frame_is_healthy(frame):
                    with self._lock:
                        self._frame = frame
                        self._frames += 1
                        self._read_ms_sum += read_ms
                        self._seq += 1
                    if self.notify is not None:
                        self.notify.set()
                elif ok:
                    self._last_error = (
                        f"/dev/video{self.index} 空帧或损坏 MJPEG，已丢弃")
                    sleep(0.005)
                else:
                    self._last_error = f"/dev/video{self.index} 读取失败"
                    sleep(0.05)
        except Exception as exc:  # 采集线程不能悄悄退出
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def seq(self) -> int:
        """已采集帧序号（拼接线程用它判断是否有新帧）。"""
        with self._lock:
            return self._seq

    def stats(self) -> tuple[int, float, str]:
        with self._lock:
            frames = self._frames
            avg_ms = self._read_ms_sum / max(frames, 1)
            error = self._last_error
            self._frames = 0
            self._read_ms_sum = 0.0
            return frames, avg_ms, error


class StitchWorker(threading.Thread):
    """拼接工作线程：任一路有新帧且双路齐备时 hstack + imencode 一次。

    预览（MJPEG 流）与感知上传共用同一份 JPEG 字节——编码从主循环
    挪到后台线程，主循环不再每帧编码，控制响应不受影响。单路掉线时
    帧序号不变，跳过重编码（防反复编码同一帧烧 CPU）。
    """

    def __init__(self, left: CameraReader, right: CameraReader,
                 notify: threading.Event, jpeg_quality: int,
                 stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="stitch-worker")
        self.left = left
        self.right = right
        self.notify = notify
        self.jpeg_quality = jpeg_quality
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._seq = 0
        self._jpeg: bytes | None = None
        self._frames = 0
        self._enc_ms_sum = 0.0

    def run(self) -> None:
        import cv2 as _cv2
        last_left = last_right = -1
        while not self.stop_event.is_set():
            # 事件驱动 + 50ms 轮询兜底（stop 检查）；事件在 clear 与
            # snapshot 之间被再次 set 时最多丢一帧，相机连续出帧自愈。
            self.notify.wait(0.05)
            self.notify.clear()
            left = self.left.snapshot()
            right = self.right.snapshot()
            if left is None or right is None:
                continue
            left_seq = self.left.seq()
            right_seq = self.right.seq()
            if left_seq == last_left and right_seq == last_right:
                continue  # 无新帧（单路掉线后事件被 set 但内容没变）
            last_left, last_right = left_seq, right_seq
            if left.shape[:2] != right.shape[:2]:
                right = _cv2.resize(right, (left.shape[1], left.shape[0]),
                                    interpolation=_cv2.INTER_AREA)
            started = monotonic()
            ok, jpeg = _cv2.imencode(
                ".jpg", np.hstack([left, right]),
                [int(_cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            enc_ms = (monotonic() - started) * 1000.0
            if not ok:
                continue
            with self._lock:
                self._seq += 1
                self._jpeg = jpeg.tobytes()
                self._frames += 1
                self._enc_ms_sum += enc_ms

    def latest(self) -> tuple[int, bytes | None]:
        """返回 (编码序号, 最新拼接 JPEG 字节)。"""
        with self._lock:
            return self._seq, self._jpeg

    def enc_stats(self) -> tuple[int, float]:
        """每秒编码统计（读后清零，1Hz 消费一次）。"""
        with self._lock:
            frames = self._frames
            avg_ms = self._enc_ms_sum / max(frames, 1)
            self._frames = 0
            self._enc_ms_sum = 0.0
            return frames, avg_ms


class StereoCameraStream:
    """两路摄像头各自独立线程采集 + 后台拼接编码线程。

    消费者：
    - ``latest_stitched()`` 取最新拼接 JPEG 字节（预览流与感知上传共用）；
    - ``stereo_frame()`` 取 BGR 拼接帧（首次 Qwen 拍照等低频场景）。
    """

    def __init__(self, config: CameraConfig, jpeg_quality: int = 70) -> None:
        devices = probe_usb_cameras(config)
        (left_index, left_node), (right_index, right_node) = devices
        if config.swap_left_right:
            left_index, right_index = right_index, left_index
            left_node, right_node = right_node, left_node
        # 分辨率/帧率自动探测（显式配置跳过）；档位变化影响主点/焦距，
        # 调用方通过 self.width/self.height 读取生效值。
        width, height, fps = probe_camera_modes(config, devices)
        effective = replace(config, width=width, height=height, fps=fps)
        self.config = effective
        self.width = width
        self.height = height
        self.left_index = left_index
        self.right_index = right_index
        self._notify = threading.Event()
        self.left = CameraReader("left", left_index, effective,
                                 notify=self._notify)
        self.right = CameraReader("right", right_index, effective,
                                  notify=self._notify)
        self._stop = threading.Event()
        self._worker = StitchWorker(self.left, self.right, self._notify,
                                    jpeg_quality, self._stop)

    def start(self) -> None:
        self.left.start()
        self.right.start()
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        self._notify.set()  # 唤醒 worker 让它检查 stop
        for reader in (self.left, self.right):
            reader.stop_event.set()
        # 不能只设置 daemon 线程的停止标志后立刻退出解释器：OpenCV
        # 仍在 cap.read()/imencode 时会触发 "terminate called without an
        # active exception"，把本应为 success/give_up 的退出码变成 -6。
        for thread in (self.left, self.right, self._worker):
            if thread.is_alive():
                thread.join(timeout=3.0)

    def latest_stitched(self) -> tuple[int, bytes | None]:
        """最新拼接 JPEG（后台线程产出，零拷贝共享）。"""
        return self._worker.latest()

    def stereo_frame(self) -> np.ndarray | None:
        """返回左右拼接 BGR 帧；任一路缺帧返回 None。"""
        left = self.left.snapshot()
        right = self.right.snapshot()
        if left is None or right is None:
            return None
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]),
                               interpolation=cv2.INTER_AREA)
        return np.hstack([left, right])

    def stats_all(self) -> dict:
        """每秒统计快照（一次取数；各子计数器读后清零）。

        日志与 pi_state 必须共用这一次调用，避免双清零。
        """
        left_frames, left_ms, left_err = self.left.stats()
        right_frames, right_ms, right_err = self.right.stats()
        enc_frames, enc_ms = self._worker.enc_stats()
        return {
            "left_fps": left_frames, "right_fps": right_frames,
            "left_ms": left_ms, "right_ms": right_ms,
            "enc_fps": enc_frames, "enc_ms": enc_ms,
            "err": left_err or right_err,
        }

    def stats_line(self) -> str:
        s = self.stats_all()
        line = (f"cam left={s['left_fps']}fps {s['left_ms']:.1f}ms"
                f" right={s['right_fps']}fps {s['right_ms']:.1f}ms"
                f" enc={s['enc_fps']}fps {s['enc_ms']:.1f}ms")
        if s["err"]:
            line += f" errors: {s['err']}"
        return line
