"""Bounded straight-start timing memory, separate from turn calibration."""
import json
import math
import os
from pathlib import Path
import time


class StartupDriveExperience:
    def __init__(self, path, binding, emit=lambda *a, **kw: None, clock=time.time):
        self.path = Path(path) if path else None
        self.binding, self.emit, self.clock = binding, emit, clock
        self.entry = None
        if self.path:
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if data['schema'] != 1 or data['binding'] != binding:
                    raise ValueError('straight-start binding/schema differs')
                self._validate(data['entry'])
                self.entry = data['entry']
                emit('startup_drive_experience_loaded', **self.entry)
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                emit('startup_drive_experience_ignored', reason=str(exc))

    def _validate(self, entry):
        if (not isinstance(entry, dict) or entry['speed_mm_s'] not in (400, 460)
                or not isinstance(entry['budget_ms'], int) or not 1800 <= entry['budget_ms'] <= 2400
                or not isinstance(entry['successes'], int) or entry['successes'] < 1
                or not math.isfinite(entry['updated_s'])
                or not 0 <= self.clock()-entry['updated_s'] <= 7*86400):
            raise ValueError('invalid/expired straight-start experience')

    def ready(self):
        try:
            self._validate(self.entry)
            return dict(self.entry)
        except (ValueError, TypeError, KeyError, OverflowError):
            return None

    def record(self, command, result, transform, reliable):
        # Requests, interrupted/expired leases, wrong direction, and jitter
        # cannot become successful startup history. Require verified movement
        # AFTER STOP too, not merely a good live ICP score.
        if (not reliable or command[0] not in (400, 460) or command[1] != 0
                or not isinstance(result, dict) or result.get('renewals', 0) < 1
                or result.get('reason') != 'measured_local_step_complete'
                or not all(math.isfinite(v) for v in
                    (transform.x_m, transform.y_m, transform.yaw_rad, result.get('elapsed_s', math.nan)))
                or not 0 < result['elapsed_s'] <= 2.4
                or transform.x_m < .015 or abs(transform.yaw_rad) > math.radians(12)):
            return
        old = self.ready() or {}
        self.entry = dict(speed_mm_s=command[0],
            budget_ms=max(1800, min(2400, math.ceil((result['elapsed_s']+.3)*1000))),
            successes=old.get('successes', 0)+1, updated_s=self.clock())
        self.emit('startup_drive_experience_learned', **self.entry)
        if self.path:
            temporary = self.path.with_name(self.path.name+'.tmp')
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(dict(schema=1, binding=self.binding,
                    entry=self.entry), indent=2), encoding='utf-8')
                os.replace(temporary, self.path)
            except OSError as exc:
                self.emit('startup_drive_experience_save_failed', reason=str(exc))
