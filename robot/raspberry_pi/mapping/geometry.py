"""二维刚体变换。"""

from dataclasses import dataclass
import math

import numpy as np


def normalize_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def transform_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        return points @ rotation.T + np.array((self.x_m, self.y_m))

    def compose(self, local_pose):
        """返回 ``self * local_pose``。"""
        translated = self.transform_points(((local_pose.x_m, local_pose.y_m),))[0]
        return Pose2D(
            float(translated[0]),
            float(translated[1]),
            normalize_angle(self.yaw_rad + local_pose.yaw_rad),
        )

    def inverse(self):
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        x = -cosine * self.x_m - sine * self.y_m
        y = sine * self.x_m - cosine * self.y_m
        return Pose2D(x, y, normalize_angle(-self.yaw_rad))

