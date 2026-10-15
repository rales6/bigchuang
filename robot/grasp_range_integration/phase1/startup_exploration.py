"""Local lidar exploration and cheap observability, not global localization.

Geometry score only decides WHEN to spend time on a map match. It never
certifies a position, clears glass, or grants motor permission by itself.
"""
import math
import numpy as np

from grasp_range_integration.phase1.rotation_guard import installed_body_mask
from raspberry_pi.mapping import Pose2D


def scan_geometry(scan):
    angles, ranges = np.asarray(scan.angles_rad, float), np.asarray(scan.distances_m, float)
    empty = dict(informative=False, reason='insufficient_geometry', points=0,
                 normal_samples=0, normal_balance=0., range_spread_m=0., sectors=0)
    if angles.ndim != 1 or angles.shape != ranges.shape:
        return dict(empty, reason='invalid_scan')
    valid = (np.isfinite(angles) & np.isfinite(ranges) & (ranges >= .25) & (ranges <= 5.)
             & ~installed_body_mask(angles, ranges))
    a, r = angles[valid], ranges[valid]
    if len(a) < 60:
        return empty
    order = np.argsort(a % (2*math.pi))
    a, r = (a[order] % (2*math.pi)), r[order]
    points = np.column_stack((r*np.cos(a), r*np.sin(a)))
    normals = []
    # Local line normals: a single long wall constrains only one direction.
    # Reject windows crossing missing rays, range jumps, and noisy corners.
    for start in range(0, len(a)-8, 3):
        aa, pp = a[start:start+9], points[start:start+9]
        if np.max(np.diff(aa)) > math.radians(4) or np.max(np.linalg.norm(np.diff(pp, axis=0), axis=1)) > .25:
            continue
        cov = np.cov(pp.T)
        values, vectors = np.linalg.eigh(cov)
        if values[1] > .001 and values[0]/values[1] < .08:
            normals.append(vectors[:, 0])
    normal_count = len(normals)
    balance = 0.
    if normal_count:
        n = np.asarray(normals)
        eigen = np.linalg.eigvalsh(n.T@n/normal_count)
        balance = float(eigen[0]/max(eigen[1], 1e-9))
    spread = float(np.percentile(r, 90)-np.percentile(r, 10))
    sectors = len(np.unique((a/(math.pi/4)).astype(int)))
    informative = normal_count >= 12 and balance >= .12 and spread >= .35 and sectors >= 3
    return dict(informative=informative,
                reason='multi_direction_structures' if informative else 'weak_or_single_direction_geometry',
                points=len(a), normal_samples=normal_count, normal_balance=balance,
                range_spread_m=spread, sectors=sectors)


def corridor_scores(scan):
    """Prefer observed clear directions; missing beams do not imply open space."""
    a, r = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
    valid = np.isfinite(a) & np.isfinite(r) & (r >= .25) & (r <= 8)
    a = (a+math.pi)%(2*math.pi)-math.pi
    scores = []
    for bearing in (-45, -30, 0, 30, 45):
        angle = math.radians(bearing)
        within = valid & (np.abs(a-angle) <= math.radians(15))
        observed = np.sort(a[within])
        supported = (len(observed) >= 12 and observed[0] <= angle-math.radians(10)
                     and observed[-1] >= angle+math.radians(10)
                     and np.max(np.diff(observed)) <= math.radians(5))
        clearance = float(np.percentile(r[within], 15)) if supported else 0.
        # Encourage a new angle without driving at a side ray directly.
        scores.append(dict(bearing_deg=bearing, supported=bool(supported),
                           clearance_m=clearance, score=min(3., clearance)))
    return scores


class ExplorationPolicy:
    def __init__(self, drive_seed=None):
        self.direction = 0
        self.arc = 0.
        self.since_fit_distance = 0.
        self.since_fit_angle = 0.
        self.total_distance = 0.
        self.low_progress = {'turn': 0, 'drive': 0}
        self.last_rate = {'turn': 0., 'drive': 0.}
        self.fit_attempts = 0
        self.since_fit_pose = Pose2D()
        self.turn_responses = {-1: dict(rate=0., low=0), 1: dict(rate=0., low=0)}
        self.drive_speed = (drive_seed or {}).get('speed_mm_s', 400)
        self.drive_budget_ms = (drive_seed or {}).get('budget_ms', 1800)

    def observe(self, transform, command):
        distance, angle = math.hypot(transform.x_m, transform.y_m), transform.yaw_rad
        self.total_distance += distance
        # Millimetre/very-small-angle fit jitter is not a new viewpoint.
        clean = Pose2D(transform.x_m if distance >= .005 else 0.,
                       transform.y_m if distance >= .005 else 0.,
                       angle if abs(angle) >= math.radians(.25) else 0.)
        self.since_fit_pose = self.since_fit_pose.compose(clean)
        self.since_fit_distance = math.hypot(self.since_fit_pose.x_m, self.since_fit_pose.y_m)
        self.since_fit_angle = self.since_fit_pose.yaw_rad
        if command[1]:
            self.arc += angle
            key, progress = 'turn', abs(angle)
            meaningful = angle*command[1] > 0 and progress >= math.radians(4)
        else:
            key, progress = 'drive', distance
            meaningful = transform.x_m > 0 and progress >= .015
        self.last_rate[key] = progress/(command[2]/1000.)
        self.low_progress[key] = 0 if meaningful else self.low_progress[key]+1
        if command[1]:
            direction = 1 if command[1] > 0 else -1
            row = self.turn_responses[direction]
            row['rate'] = self.last_rate[key]
            row['low'] = 0 if meaningful else row['low']+1
        elif meaningful:
            self.drive_speed = max(400, command[0])

    def wants_fit(self, geometry):
        new_view = abs(self.since_fit_angle) >= math.radians(15) or self.since_fit_distance >= .12
        # One full-heading check of rich INITIAL evidence is free of motor
        # permission. Only retries require a genuinely different viewpoint.
        return geometry['informative'] and (self.fit_attempts == 0 or new_view)

    def continuous_drive_budget(self):
        return min(2400, self.drive_budget_ms+300*self.low_progress['drive'])

    def fitted(self):
        self.since_fit_angle = self.since_fit_distance = 0.
        self.since_fit_pose = Pose2D()
        self.fit_attempts += 1

    def candidates(self, scan):
        scores = corridor_scores(scan)
        front = next(v for v in scores if v['bearing_deg'] == 0)
        best = max(scores, key=lambda v:v['score'])
        # Commit to a useful arc, not left/right cancellation after a few
        # degrees. Select a new direction only after ~25deg or blocked output.
        if not self.direction or abs(self.arc) >= math.radians(25):
            self.direction = 1 if best['bearing_deg'] > 0 else -1
            self.arc = 0.
        def turn_ttl(direction):
            row = self.turn_responses[direction]
            measured = int(math.radians(8)/row['rate']*1000) if row['rate'] > .01 else 180
            return min(400, max(180, measured)+60*row['low'])
        # This is an individual firmware lease, NOT the breakaway time.
        # Fresh approved scans renew identical leases without issuing STOP.
        drive_ttl = 400
        drive_speed = 460 if self.low_progress['drive'] >= 2 else self.drive_speed
        turn = (0, self.direction*3500, turn_ttl(self.direction))
        opposite = (0, -self.direction*3500, turn_ttl(-self.direction))
        drive = (drive_speed, 0, drive_ttl)
        # A few degrees of noise must not reset the breakaway adaptation.
        # Geometry is informative? translate for a genuinely different view;
        # otherwise pursue the more open measured corridor first.
        prefer_drive = (front['supported'] and front['clearance_m'] >= .9
                        and (best['bearing_deg'] == 0 or abs(self.arc) >= math.radians(20)))
        drives = [drive] if drive_speed == 400 else [drive, (400, 0, drive_ttl)]
        return (drives+[turn, opposite] if prefer_drive else [turn]+drives+[opposite]), scores
