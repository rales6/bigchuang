"""Small inverse-kinematics helper for the legacy 4-servo arm geometry."""

from math import acos, atan, cos, pi, sin, sqrt


class ArmKinematics:
    def __init__(self, l0=170, l1=105, l2=75, l3=185):
        self.l0 = l0 * 10
        self.l1 = l1 * 10
        self.l2 = l2 * 10
        self.l3 = l3 * 10

    def solve(self, x_mm, y_mm, z_mm, duration_ms):
        best = None
        for alpha in range(-25, -65, -1):
            pulses = self._solve_alpha(x_mm, y_mm, z_mm, alpha)
            if pulses is not None:
                best = pulses
        if best is None:
            return None
        return [(joint_id, best[joint_id], duration_ms) for joint_id in range(4)]

    def from_reach_yaw(self, reach_mm, yaw_deg, z_mm, duration_ms):
        yaw_rad = yaw_deg * pi / 180.0
        y_mm = reach_mm * cos(yaw_rad)
        x_mm = reach_mm * sin(yaw_rad)
        return self.solve(x_mm, y_mm, z_mm, duration_ms)

    def _solve_alpha(self, x, y, z, alpha):
        x *= 10
        y *= 10
        z *= 10
        if y < 0:
            return None

        if x == 0 and y != 0:
            theta6 = 0.0
        elif x > 0 and y == 0:
            theta6 = 90.0
        elif x < 0 and y == 0:
            theta6 = -90.0
        else:
            theta6 = atan(x / y) * 180.0 / pi

        reach = sqrt(x * x + y * y)
        wrist_y = reach - self.l3 * cos(alpha * pi / 180.0)
        wrist_z = z - self.l0 - self.l3 * sin(alpha * pi / 180.0)
        distance = sqrt(wrist_y * wrist_y + wrist_z * wrist_z)
        if wrist_z < -self.l0 or distance > self.l1 + self.l2:
            return None

        bbb = (
            wrist_y * wrist_y + wrist_z * wrist_z + self.l1 * self.l1 -
            self.l2 * self.l2
        ) / (2 * self.l1 * distance)
        aaa = -(
            wrist_y * wrist_y + wrist_z * wrist_z - self.l1 * self.l1 -
            self.l2 * self.l2
        ) / (2 * self.l1 * self.l2)
        if bbb < -1 or bbb > 1 or aaa < -1 or aaa > 1:
            return None

        z_sign = -1 if wrist_z < 0 else 1
        theta5 = (acos(wrist_y / distance) * z_sign + acos(bbb)) * 180.0 / pi
        theta4 = 180.0 - acos(aaa) * 180.0 / pi
        theta3 = alpha - theta5 + theta4
        if theta5 > 180 or theta5 < 0:
            return None
        if theta4 > 135 or theta4 < -135:
            return None
        if theta3 > 90 or theta3 < -90:
            return None

        return (
            int(1500 - 2000.0 * theta6 / 270.0),
            int(1500 + 2000.0 * (theta5 - 90.0) / 270.0),
            int(1500 + 2000.0 * theta4 / 270.0),
            int(1500 + 2000.0 * theta3 / 270.0),
        )
