# -*- coding: utf-8 -*-
"""采集改造的单元测试：StitchWorker（单次编码共享、无新帧不重编码、
单路缺帧不产出、统计清零语义）与 probe_camera_modes 的显式跳过逻辑。

无摄像头也可运行（不打开真实设备）；需要 cv2/numpy。
运行：python pursuit/test_camera_stream.py（track 目录下）
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

import cv2
import numpy as np

from pursuit.camera_stream import StitchWorker, probe_camera_modes
from pursuit.pursuit_config import CameraConfig


class FakeReader:
    """模拟 CameraReader 的桩：set_frame 推进 seq，可缺帧。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0

    def set_frame(self, array: np.ndarray) -> None:
        with self._lock:
            self._frame = array
            self._seq += 1

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def seq(self) -> int:
        with self._lock:
            return self._seq


def _wait_latest(worker: StitchWorker, target_seq: int,
                 timeout_s: float = 3.0) -> tuple[int, bytes | None]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        seq, jpeg = worker.latest()
        if seq >= target_seq:
            return seq, jpeg
        time.sleep(0.02)
    raise AssertionError(f"{timeout_s}s 内未等到 seq>={target_seq} "
                         f"(当前 {worker.latest()[0]})")


def test_stitch_worker() -> None:
    left, right = FakeReader(), FakeReader()
    notify = threading.Event()
    stop = threading.Event()
    worker = StitchWorker(left, right, notify, jpeg_quality=70, stop_event=stop)
    worker.start()
    try:
        # 1) 双路缺帧：不产出
        notify.set()
        time.sleep(0.15)
        assert worker.latest()[0] == 0, "缺帧不应产出"

        # 2) 单路有帧、另一路缺：仍不产出（拼接需要两路）
        left.set_frame(np.zeros((240, 640, 3), np.uint8) + 10)
        notify.set()
        time.sleep(0.15)
        assert worker.latest()[0] == 0, "单路缺帧不应产出"

        # 3) 双路齐备：产出拼接 JPEG（1280 宽）
        right.set_frame(np.zeros((240, 640, 3), np.uint8) + 20)
        notify.set()
        seq, jpeg = _wait_latest(worker, 1)
        assert jpeg is not None and len(jpeg) > 1000, "拼接 JPEG 过小"
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape == (240, 1280, 3), \
            f"拼接尺寸错误: {image.shape if image is not None else None}"

        # 4) 无新帧：seq 不推进（防单路掉线反复重编码同帧烧 CPU）
        time.sleep(0.15)
        assert worker.latest()[0] == seq, "无新帧时 seq 不应变化"

        # 5) 单路更新（另一路陈旧）：允许产出（拼接用各自最新帧）
        left.set_frame(np.zeros((240, 640, 3), np.uint8) + 30)
        notify.set()
        seq2, _ = _wait_latest(worker, seq + 1)
        assert seq2 == seq + 1

        # 6) 双路同时更新：继续推进
        left.set_frame(np.zeros((240, 640, 3), np.uint8) + 40)
        right.set_frame(np.zeros((240, 640, 3), np.uint8) + 50)
        notify.set()
        seq3, _ = _wait_latest(worker, seq2 + 1)
        assert seq3 == seq2 + 1

        # 7) enc_stats 读后清零（1Hz 统计消费一次语义）。
        # 注意：Windows 上 monotonic() 分辨率可能只有 ~15.6ms，小图编码
        # 耗时被量化为 0——avg_ms 只做展示，核心断言是计数与清零。
        frames, avg_ms = worker.enc_stats()
        assert frames >= 3, f"编码统计异常: {frames}"
        frames2, avg_ms2 = worker.enc_stats()
        assert frames2 == 0 and avg_ms2 == 0.0, "enc_stats 未清零"
        print(f"[test_camera_stream] StitchWorker OK "
              f"(encoded={frames} avg={avg_ms:.1f}ms)")
    finally:
        stop.set()
        notify.set()
        worker.join(timeout=2)


def test_probe_skip_when_explicit() -> None:
    """显式配置时跳过档位探测（不打开摄像头）。"""
    cases = [
        (CameraConfig(left_device=0, right_device=2), (1280, 720, 30),
         "指定摄像头编号应跳过探测"),
        (CameraConfig(auto_probe=False), (1280, 720, 30),
         "auto_probe=False 应跳过探测"),
        (CameraConfig(width=960, height=540), (960, 540, 30),
         "非默认分辨率应跳过探测"),
        (CameraConfig(fps=15), (1280, 720, 15),
         "非默认 fps 应跳过探测"),
    ]
    for config, expected, message in cases:
        got = probe_camera_modes(config, [])
        assert got == expected, f"{message}: got {got}"
    print("[test_camera_stream] probe 显式跳过逻辑 OK")


if __name__ == "__main__":
    test_probe_skip_when_explicit()
    test_stitch_worker()
    print("ALL_CAMERA_STREAM_TESTS_PASSED")
