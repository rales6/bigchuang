"""Offline scan availability/ownership checks; no serial or motion access."""
import threading
import time
import types
import unittest
from unittest.mock import Mock

from grasp_range_integration.track_web_integration.pi.pursuit_scan_recovery import reacquire_scan


class Driver:
    def __init__(self, stamps):
        self.stamps = stamps
        self.closed = False

    def scans(self):
        for stamp in self.stamps:
            yield types.SimpleNamespace(timestamp_s=stamp, angles_rad=[0.]*32,
                                        distances_m=[1.]*32)

    def close(self):
        self.closed = True


class ScanRecoveryTests(unittest.TestCase):
    def call(self, driver, cancel=None):
        return reacquire_scan(cancel or threading.Event(), emit=Mock(),
                              driver_factory=lambda: driver, timeout_s=.1)

    def test_two_distinct_fresh_scans_release_driver(self):
        now = time.monotonic()
        driver = Driver([now-.08, now-.04])
        self.assertTrue(self.call(driver)['ok'])
        self.assertTrue(driver.closed)

    def test_stale_future_duplicate_empty_and_nonfinite_rejected(self):
        now = time.monotonic()
        for stamps in [[], [now-2, now-1], [now+2, now+3], [now, now], [float('nan')]*2]:
            driver = Driver(stamps)
            with self.subTest(stamps=stamps), self.assertRaises(RuntimeError):
                self.call(driver)
            self.assertTrue(driver.closed)

    def test_cancel_does_not_open_serial(self):
        cancel = threading.Event(); cancel.set()
        factory = Mock()
        with self.assertRaises(RuntimeError):
            reacquire_scan(cancel, emit=Mock(), driver_factory=factory)
        factory.assert_not_called()

    def test_reader_error_releases_driver_and_does_not_authorize(self):
        driver = Driver([])
        driver.scans = Mock(side_effect=OSError('serial lost'))
        with self.assertRaises(RuntimeError):
            self.call(driver)
        self.assertTrue(driver.closed)

    def test_cancel_during_read_releases_driver(self):
        cancel = threading.Event()
        driver = Driver([])
        def scans():
            cancel.set()
            yield types.SimpleNamespace(timestamp_s=time.monotonic(), angles_rad=[], distances_m=[])
        driver.scans = scans
        with self.assertRaises(RuntimeError):
            self.call(driver, cancel)
        self.assertTrue(driver.closed)


if __name__ == '__main__':
    unittest.main()
