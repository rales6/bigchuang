"""Small absolute model-pose corrections, applied once before claw close.

S0 correction was removed at the operator's request on 2026-10-03.
Firmware mirrors S3 from S2. Placement reuses the corrected pose and must
not call this function again. Pulse offsets are NOT calibrated degrees.
"""
import os


def corrected_grasp_pose(target, servos, environ=None):
    environ = os.environ if environ is None else environ
    corrected = dict(target)
    changed = False
    # Ignore even a stale S0 environment override: keep the model's S0.
    for key, limit, default in (("s2", 25, "20"),):
        name = "ARM_GRASP_" + key.upper() + "_NUDGE_US"
        raw = environ.get(name, default)
        try:
            delta = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(name + " must be an integer") from exc
        if not -limit <= delta <= limit:
            raise ValueError(f"{name} must be within {-limit}..{limit} us")
        if delta:
            spec = next(item for item in servos if item["key"] == key)
            desired = int(target[key]) + delta
            if not spec["min"] <= desired <= spec["max"]:
                raise ValueError(key + " nudge would cross the configured servo limit")
            corrected[key] = desired
            changed = True
    return corrected if changed else None
