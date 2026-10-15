"""Stationary scan reacquisition between two pursuit children; no chassis IO."""
import math
import threading
import time


def reacquire_scan(cancelled, *, port=None, timeout_s=12., emit, driver_factory=None):
    """Open only after the previous child exits; release before the next one.

    This establishes scan availability, not rear clearance or a movement
    permit. The fresh pursuit controller still checks each actual command.
    """
    if cancelled.is_set():
        raise RuntimeError('任务已取消，未重新获取雷达扫描')
    if driver_factory is None:
        from raspberry_pi.config import LidarConfig
        from raspberry_pi.lidar.n10_driver import N10LidarDriver
        driver_factory = lambda: N10LidarDriver(LidarConfig(**({'port': port} if port else {})))
    driver = driver_factory()
    done = threading.Event()
    stop = threading.Event()
    evidence = {}

    def read():
        try:
            streak = 0
            previous = None
            for scan in driver.scans():
                if stop.is_set() or cancelled.is_set():
                    break
                now = time.monotonic()
                stamp = float(scan.timestamp_s)
                valid = sum(math.isfinite(float(a)) and math.isfinite(float(d)) and 0 < float(d) <= 8
                            for a, d in zip(scan.angles_rad, scan.distances_m))
                fresh = math.isfinite(stamp) and 0 <= now-stamp <= .5
                streak = streak+1 if fresh and valid >= 16 and (previous is None or stamp > previous) else 0
                previous = stamp
                if streak >= 2:
                    evidence.update(ok=True, scan_age_s=now-stamp, valid_points=valid)
                    break
        except Exception as exc:
            if not stop.is_set():
                evidence['error'] = str(exc)
        finally:
            done.set()

    reader = threading.Thread(target=read, name='pursuit-scan-reacquire', daemon=True)
    reader.start()
    deadline = time.monotonic()+timeout_s
    try:
        while not done.wait(.1):
            if cancelled.is_set() or time.monotonic() >= deadline:
                break
    finally:
        stop.set()
        driver.close()
        reader.join(timeout=1.)
    if cancelled.is_set():
        raise RuntimeError('任务已取消，未启动新追踪')
    if reader.is_alive() or evidence.get('ok') is not True:
        emit('pursuit_scan_reacquire_pending', error=evidence.get('error', 'no_fresh_scans'))
        raise RuntimeError('靠近恢复尚无两圈新雷达扫描，保持待恢复；未执行盲目回退')
    emit('pursuit_scan_reacquired', **evidence)
    return evidence
