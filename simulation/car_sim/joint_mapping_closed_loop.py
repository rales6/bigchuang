"""Production mapping loop with synthetic PGM lidar and a bounded chassis model.

Unlike prescribed-route replay, commands from FrontierExplorer advance the car.
Truth is used only by the sensor/plant and the evaluator, never for localization.
"""
import argparse
import contextlib
import hashlib
import math
from pathlib import Path

import numpy as np

from car_sim.joint_mapping_replay import raycast
from raspberry_pi import autonomous_mapping
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.lidar import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.joint_mapping import atomic_json, angle_distance


class SimulatedCar:
    active_transport = "simulated_ble"
    def __init__(self, known, seed, steps):
        self.known = known
        self.rng = np.random.default_rng(seed)
        self.limit = steps
        self.pose = Pose2D(-1, .65, -math.pi / 2)
        self.origin = self.pose
        self.now = 1.
        self.linear = self.angular = 0.
        self.expiry = 0.
        self.collision = False
        self.frames = []
        self.views = []
        self.max_error = self.max_yaw = 0.
        self.distance = 0.
        self.finished = False

    def heartbeat(self):
        pass

    def set_twist(self, linear, angular, ttl_ms):
        # Not a calibrated motor model. Model loaded speed below requested
        # controller units, but retain dead zone and per-frame noise.
        self.linear = 0 if abs(linear) < 150 else linear * .00048
        self.angular = 0 if abs(angular) < 700 else angular * .00030
        self.expiry = self.now + ttl_ms / 1000

    def stop(self):
        self.linear = self.angular = 0

    def close(self):
        self.stop()

    def scans(self):
        for i in range(self.limit):
            dt = min(.2, max(0., self.expiry - self.now))
            v = self.linear * self.rng.uniform(.97, 1.03)
            w = self.angular * self.rng.uniform(.97, 1.03)
            previous = self.pose
            for fraction in np.linspace(.1, 1, 10):
                t = dt * float(fraction)
                yaw = previous.yaw_rad + w * t
                if abs(w) > 1e-8:
                    x = previous.x_m + v / w * (math.sin(yaw) - math.sin(previous.yaw_rad))
                    y = previous.y_m - v / w * (math.cos(yaw) - math.cos(previous.yaw_rad))
                else:
                    x = previous.x_m + v * t * math.cos(yaw)
                    y = previous.y_m + v * t * math.sin(yaw)
                col, row = self.known.world_to_grid(((x, y),))[0]
                if not (self.known.in_bounds(col, row) and self.known.traversable[row, col]):
                    self.collision = True
                    raise RuntimeError("simulation swept footprint intersects obstacle/unknown")
            self.pose = Pose2D(x, y, yaw)
            self.distance += math.hypot(x - previous.x_m, y - previous.y_m)
            self.now += .2
            angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
            distances = raycast(self.known, self.pose, angles)
            distances += self.rng.normal(0, .008, len(angles))
            distances[self.rng.random(len(angles)) < .02] = np.nan
            yield LaserScan(angles, distances, self.now)

    def record_scan(self, scan, update, command):
        truth = self.origin.inverse().compose(self.pose)
        error = math.hypot(update.pose.x_m - truth.x_m, update.pose.y_m - truth.y_m)
        yaw_error = math.degrees(angle_distance(update.pose.yaw_rad, truth.yaw_rad))
        self.max_error = max(self.max_error, error)
        self.max_yaw = max(self.max_yaw, yaw_error)
        self.frames.append({"timestamp": self.now, "truth": [truth.x_m, truth.y_m, truth.yaw_rad],
                            "estimated": [update.pose.x_m, update.pose.y_m, update.pose.yaw_rad],
                            "error_m": error, "yaw_error_deg": yaw_error,
                            "state": command.state if command else "startup"})
        if len(self.frames) % 100 == 0:
            print("SIM frames={} travel={:.2f}m max_error={:.3f}m max_yaw={:.1f}deg".format(len(self.frames), self.distance, self.max_error, self.max_yaw))
        if error > .4 or yaw_error > 20:
            raise RuntimeError("simulation localization error exceeded acceptance envelope")

    def due(self, pose):
        return not any(math.hypot(pose.x_m - x, pose.y_m - y) < .65
                       and angle_distance(pose.yaw_rad, yaw) < math.radians(55)
                       for x, y, yaw in self.views)

    def capture(self, scan, update, grid):
        assert self.linear == 0 and self.angular == 0, "capture while moving"
        self.views.append((update.pose.x_m, update.pose.y_m, update.pose.yaw_rad))
        self.now += 2  # delayed camera; TTL cannot keep old motion alive

    def finish_map(self, grid, finished=False):
        self.finished = bool(finished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="simulation_runs/joint_closed_loop")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = ["raspberry_pi/autonomous_mapping.py", "raspberry_pi/joint_mapping.py", "raspberry_pi/config.py"]
    sources += [p.relative_to(root).as_posix() for p in (root / "raspberry_pi/mapping").glob("*.py")]
    source_hashes = {p: hashlib.sha256((root / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sources}
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    known = KnownOccupancyMap.load("maps/ble_scene_17.pgm", clearance_m=.20)
    sim = SimulatedCar(known, args.seed, args.steps)
    options = autonomous_mapping.build_argument_parser().parse_args([
        "--enable-motion", "--lidar-offset-x", "0", "--output", str(directory / "map"),
        "--mapping-max-distance", "4", "--max-runtime-min", "15",
        "--speed", "250", "--max-drive-speed", "420", "--max-turn-speed", "2600",
    ])
    error = None
    try:
        with (directory / "controller.log").open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
            autonomous_mapping.run(options, client=sim, lidar=sim,
                                   clock=lambda: sim.now, mapping_observer=sim)
    except Exception as exc:
        error = str(exc)
    result = {"kind": "production_controller_and_slam_synthetic_pgm_lidar",
              "source_sha256": source_hashes,
              "limitations": "Uncalibrated circular 0.20 m chassis, no glass/rolling scan/communication dropouts; not field validation.",
              "seed": args.seed, "frames": len(sim.frames), "distance_m": sim.distance,
              "max_error_m": sim.max_error, "max_yaw_error_deg": sim.max_yaw,
              "collision": sim.collision, "error": error,
              "completed_before_step_limit": error is None and sim.finished,
              "views": sim.views, "trajectory": sim.frames}
    result["passed"] = (result["completed_before_step_limit"] and sim.distance > 5
                        and sim.max_error < .2 and sim.max_yaw < 10 and not sim.collision)
    atomic_json(directory / "report.json", result)
    print({k: v for k, v in result.items() if k not in ("trajectory", "views")})
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
