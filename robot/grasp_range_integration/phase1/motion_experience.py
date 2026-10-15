"""Real angular progress deadlines and narrowly bound, durable turn experience.

No map pose, clock, frozen mission or obstacle verdict is altered here.
The deadline implements the mission's existing addition/comparison protocol.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time


class AngularProgress:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.key = None
        self.best = None
        self.physical = 0.
        self.last_progress = None
        self.suspended_intervals = []

    def suspend(self, started, finished):
        if all(math.isfinite(v) for v in (started, finished)) and finished >= started:
            self.suspended_intervals.append((started, finished))

    def suspended_since(self, started, now):
        # Union overlapping stationary + active-recovery intervals so nested
        # recovery does not grant the same excluded time twice.
        intervals = sorted((max(a, started), min(b, now)) for a,b in self.suspended_intervals
                           if b > started and a < now)
        total, end = 0., started
        for a,b in intervals:
            total += max(0., b-max(a,end))
            end = max(end,b)
        return total

    def feedback(self, rotation, error, reliable):
        if (reliable and error is not None and math.isfinite(rotation)
                and .25 <= abs(math.degrees(rotation)) <= 24):
            # Wrong-way motion erases accumulated evidence, never grants time.
            self.physical = max(0., self.physical + rotation * (1 if error > 0 else -1))

    def heading(self, key, error):
        if key != self.key:
            self.key, self.best = key, abs(error)
            self.physical = 0.
            self.last_progress = None
            return
        reduction = self.best - abs(error)
        threshold = math.radians(2)
        if reduction >= threshold and self.physical >= threshold:
            self.best = abs(error)
            self.physical = 0.
            self.last_progress = self.clock()
            from grasp_range_integration.phase1.turn_policy import emit
            emit("navigation_angular_progress", remaining_deg=math.degrees(error),
                 deadline_refresh_seconds=35, physical_and_heading_confirmed=True)


class ProgressTimeout:
    """XY refresh remains original; verified yaw also refreshes the lease.

    Blocked/stalled/unreliable turns still expire after 35s. A waypoint with
    only angular progress cannot consume more than 120s without XY progress.
    """
    def __init__(self, progress, seconds=35., maximum=120.):
        self.progress, self.seconds, self.maximum = progress, seconds, maximum

    def __radd__(self, now):
        return ProgressDeadline(float(now), self)


class ProgressDeadline:
    def __init__(self, created, policy):
        self.created, self.policy = created, policy

    def __le__(self, now):
        policy = self.policy
        last = policy.progress.last_progress
        refreshed = max(self.created, last if last is not None else self.created)
        deadline = min(self.created + policy.maximum + policy.progress.suspended_since(self.created, float(now)),
                       refreshed + policy.seconds + policy.progress.suspended_since(refreshed, float(now)))
        return float(now) >= deadline


class TurnExperience:
    """Only learn the already calibrated 3500/180ms maximum-output pulse.

    Other commands are not extrapolated. Directions are independent; a fast
    response is retained to SHORTEN later leases. Two actual successes are
    required before reuse. Disk profiles expire after seven days and bind to
    the adapter/calibration/body-mask code, link identity and output balance.
    """
    def __init__(self, path, binding, emit=lambda *a, **k: None, clock=time.time,
                 compatible_bindings=()):
        self.path = Path(path) if path else None
        self.binding, self.emit, self.clock = binding, emit, clock
        self.entries = {}
        self.disabled = set()
        if self.path:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if (not isinstance(data, dict) or data.get("schema") != 1
                        or data.get("binding") not in {binding,*compatible_bindings}
                        or not isinstance(data.get("directions"), dict)):
                    raise ValueError("profile binding/schema differs")
                for direction, entry in data["directions"].items():
                    self._validate(direction, entry)
                self.entries = data["directions"]
                if data["binding"] != binding:
                    emit("turn_experience_binding_migrated", previous_binding=data["binding"],
                         validated_pulse_contract_unchanged=True)
                emit("turn_experience_loaded", directions=list(self.entries), path=str(self.path))
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                emit("turn_experience_ignored", reason=str(exc))

    def _validate(self, direction, entry):
        if not isinstance(entry, dict):
            raise ValueError("invalid direction entry")
        stamp, rate = float(entry["updated_s"]), float(entry["fastest_rate_rad_s"])
        if (direction not in {"left", "right"} or entry["ttl_ms"] != 180
                or entry["speed_mrad_s"] != (3500 if direction == "left" else -3500)
                or not isinstance(entry["successes"], int) or entry["successes"] < 1
                or not math.isfinite(rate) or not 0 < rate <= math.radians(24)/.180
                or not math.isfinite(stamp) or not 0 <= self.clock()-stamp <= 7*86400):
            raise ValueError("invalid/expired turn experience")

    def ready(self, error):
        direction = "left" if error > 0 else "right"
        entry = self.entries.get(direction)
        if direction in self.disabled or entry is None:
            return None
        try:
            self._validate(direction, entry)
        except (ValueError, KeyError, TypeError):
            return None
        return entry if entry["successes"] >= 2 else None

    def record(self, pulse, rotation, translation, reliable):
        if not pulse:
            return
        speed = pulse.get("angular_mrad_s", 0)
        direction = "left" if speed > 0 else "right"
        if (not reliable or abs(speed) != 3500 or pulse.get("ttl_ms") != 180
                or not math.isfinite(rotation) or not math.isfinite(translation)
                or rotation*speed <= 0 or not .25 <= abs(math.degrees(rotation)) <= 24
                or abs(translation) > .03):
            return
        # Must reproduce the actual tested wheel encoding, not just request.
        expected = (1.5, 1.5/1.15, .9*1.5/1.15, 1.5/1.15)
        gains = pulse.get("wheel_output_gains", ())
        if len(gains) != 4 or any(not math.isfinite(float(a)) or abs(float(a)-b) > 1e-5
                                  for a, b in zip(gains, expected)):
            return
        old = self.entries.get(direction, {})
        try:
            self._validate(direction, old)
        except (ValueError, KeyError, TypeError):
            old = {}
        stamp = float(pulse.get("observed_s", self.clock()))
        if not math.isfinite(stamp) or not 0 <= self.clock()-stamp <= 7*86400:
            return
        self.entries[direction] = dict(speed_mrad_s=speed, ttl_ms=180,
            successes=int(old.get("successes", 0))+1, updated_s=stamp,
            fastest_rate_rad_s=max(old.get("fastest_rate_rad_s", 0), abs(rotation)/.180))
        self.emit("turn_experience_learned", direction=direction, **self.entries[direction])
        if self.path:
            temporary = self.path.with_name(self.path.name + ".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(dict(schema=1, binding=self.binding,
                    directions=self.entries), indent=2), encoding="utf-8")
                os.replace(temporary, self.path)
            except OSError as exc:
                self.emit("turn_experience_save_failed", reason=str(exc))

    def import_audited_history(self, events):
        """Import acknowledged pulses paired with valid stopped-scan feedback.

        Used only for the previously inspected 20260927_162135 run, not an
        arbitrary directory's requested speeds or final controller state.
        """
        pulse, window = None, None
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("invalid history event")
            kind = event.get("type")
            if kind == "turn_motor_output_boost":
                pulse, window = event, None
            elif kind in {"turn_start_failed", "turn_command_ack_failed", "motion_interlock_fault",
                          "turn_feedback_fault", "command", "turn_command_ack"}:
                pulse, window = None, None
            elif kind == "turn_feedback_window":
                window = event if event.get("valid") is True else None
            elif kind == "measured_motion":
                if (pulse is not None and window is not None
                        and window.get("reason") == "post_pulse_fresh_scan"
                        and event.get("reliable") is True
                        and math.isfinite(float(event.get("time", math.nan)))
                        and 0 <= event["time"] - pulse["time"] <= 5
                        and abs(window["measured_rotation_deg"] - event["rotation_deg"]) < .01):
                    self.record(dict(pulse, observed_s=event["time"]),
                        math.radians(event["rotation_deg"]), event["translation_m"], True)
                pulse, window = None, None


def configure_motion_experience(mission, args):
    from grasp_range_integration.phase1.turn_policy import emit
    base = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    # Arrival/timing implementation edits do not invalidate physically
    # unchanged pulse evidence. Calibration/output/body-mask/identity do.
    digest.update(b"calibrated-start-contract-v1")
    for name in ("turn_calibration.py", "turn_output_boost.py", "rotation_guard.py"):
        digest.update((base/name).read_bytes())
    identity = {name: getattr(args, name, None) for name in (
        "link", "ble_name", "ble_address", "esp_port", "lidar_port",
        "front_left_gain", "front_right_gain", "rear_left_gain", "rear_right_gain")}
    digest.update(json.dumps(identity, sort_keys=True).encode())
    path = base/"runtime"/"motion_experience.json"
    audited_identity = dict(link="ble",ble_name="ESP32-Robot-Car",ble_address="64:B7:08:69:9A:E6",
        esp_port="/dev/serial0",lidar_port="/dev/ttyACM0",front_left_gain=1.15,
        front_right_gain=1.,rear_left_gain=.9,rear_right_gain=1.)
    compatible = ("3612fdb8f0e89a5171b046470fd4c8a42c91455bf7f540055f1e74540d9a518a",) if identity == audited_identity else ()
    experience = mission._phase1_turn_experience = TurnExperience(
        path, digest.hexdigest(), emit, compatible_bindings=compatible)
    from grasp_range_integration.phase1.startup_drive_experience import StartupDriveExperience
    drive_binding = hashlib.sha256((digest.hexdigest()+':straight-start-v1').encode()).hexdigest()
    mission._phase1_startup_drive_experience = StartupDriveExperience(
        base/'runtime'/'startup_drive_experience.json', drive_binding, emit)
    gains = mission._serial_config(args).wheel_output_gains
    if (not path.exists() and identity["link"] == "ble"
            and identity["ble_name"] == "ESP32-Robot-Car"
            and len(gains) == 4
            and all(abs(float(a)-b) < 1e-8 for a, b in zip(gains, (1.15, 1., .9, 1.)))):
        history = Path(args.output)/"20260927_162135"/"phase1_motion.jsonl"
        try:
            if history.stat().st_size > 20*1024*1024:
                raise ValueError("history too large")
            events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
            experience.import_audited_history(events)
            emit("turn_experience_history_imported", source=str(history), directions=list(experience.entries))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            emit("turn_experience_history_ignored", reason=str(exc))
    args.navigation_no_progress_seconds = ProgressTimeout(mission._phase1_angular_progress)
