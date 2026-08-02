"""带粗角度初始化和稳健裁剪的轻量级二维 ICP 扫描匹配。"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import Pose2D


@dataclass(frozen=True)
class MatchResult:
    transform: Pose2D
    rmse_m: float
    correspondences: int
    converged: bool
    inlier_ratio: float = 0.0


class IcpScanMatcher:
    def __init__(
        self,
        max_correspondence_m=0.35,
        min_points=35,
        max_iterations=18,
        coarse_yaw_range_deg=60.0,
        coarse_yaw_step_deg=4.0,
        trim_fraction=0.85,
    ):
        self.max_correspondence_m = float(max_correspondence_m)
        self.min_points = int(min_points)
        self.max_iterations = int(max_iterations)
        self.coarse_yaw_range_deg = float(coarse_yaw_range_deg)
        self.coarse_yaw_step_deg = float(coarse_yaw_step_deg)
        self.trim_fraction = float(trim_fraction)

    def match(
        self,
        reference_points,
        current_points,
        initial=None,
        coarse_yaw_range_deg=None,
    ):
        reference = np.asarray(reference_points, dtype=np.float64)
        current = np.asarray(current_points, dtype=np.float64)
        if len(reference) < self.min_points or len(current) < self.min_points:
            estimate = initial or Pose2D()
            return MatchResult(estimate, math.inf, 0, False, 0.0)

        if initial is not None:
            estimate = initial
        elif self._identity_is_reliable(reference, current):
            # Only skip coarse yaw search when virtually the complete scan
            # overlaps with very small residual error. The former 75% gate
            # also classified an 8-degree rectangular-room turn as identity.
            estimate = Pose2D()
        else:
            # A rotated long wall can still have excellent nearest-neighbour
            # overlap at identity. Starting ICP at zero then converges to an
            # along-wall local minimum and accumulates yaw bias every frame.
            estimate = self._coarse_initial(
                reference,
                current,
                yaw_range_deg=coarse_yaw_range_deg,
            )
        previous_rmse = math.inf
        converged = False

        for _ in range(self.max_iterations):
            transformed = estimate.transform_points(current)
            indices, squared_distances = self._nearest(reference, transformed)
            accepted = squared_distances <= self.max_correspondence_m ** 2
            count = int(np.count_nonzero(accepted))
            if count < self.min_points:
                return MatchResult(estimate, math.inf, count, False, self._ratio(
                    count, reference, current
                ))

            accepted_indices = np.flatnonzero(accepted)
            # 去掉距离最大的部分对应点，避免少量孤立远点拉偏刚体变换。
            keep_count = max(
                self.min_points,
                int(math.ceil(count * self.trim_fraction)),
            )
            if keep_count < count:
                order = np.argsort(squared_distances[accepted_indices])
                accepted_indices = accepted_indices[order[:keep_count]]

            source = transformed[accepted_indices]
            target = reference[indices[accepted_indices]]
            delta = self._rigid_transform(source, target)
            estimate = delta.compose(estimate)
            rmse = float(np.sqrt(np.mean(
                squared_distances[accepted_indices]
            )))
            if (
                abs(previous_rmse - rmse) < 1e-4
                and math.hypot(delta.x_m, delta.y_m) < 2e-4
                and abs(delta.yaw_rad) < 2e-4
            ):
                converged = True
                break
            previous_rmse = rmse

        rmse, count = self._evaluate(reference, current, estimate)
        return MatchResult(
            estimate,
            rmse,
            count,
            converged or math.isfinite(rmse),
            self._ratio(count, reference, current),
        )

    def _coarse_initial(
        self,
        reference,
        current,
        yaw_range_deg=None,
    ):
        """在较大角度范围内寻找 ICP 初值，避免高速旋转落入错误局部极值。"""
        reference_coarse = self._uniform_sample(reference, 140)
        current_coarse = self._uniform_sample(current, 140)
        coarse_limit = max(self.max_correspondence_m * 2.0, 0.70)
        coarse_limit_squared = coarse_limit ** 2
        yaw_range = abs(
            self.coarse_yaw_range_deg
            if yaw_range_deg is None
            else float(yaw_range_deg)
        )
        yaw_step = max(0.5, abs(self.coarse_yaw_step_deg))
        candidates = np.arange(
            -yaw_range,
            yaw_range + yaw_step * 0.5,
            yaw_step,
        )

        best_estimate = Pose2D()
        best_score = (-math.inf, -math.inf, -math.inf)
        for yaw_deg in candidates:
            estimate = Pose2D(yaw_rad=math.radians(float(yaw_deg)))
            transformed = estimate.transform_points(current_coarse)
            _indices, squared = self._nearest(reference_coarse, transformed)
            accepted = squared <= coarse_limit_squared
            count = int(np.count_nonzero(accepted))
            if count < self.min_points:
                continue

            # 粗搜索只确定角度，不在部分视场上用质心/最近邻估算平移。
            # 否则旋转后新进入视场的墙面会被误判为大幅车辆平移。
            if count:
                values = np.sort(squared[accepted])
                keep = max(1, int(math.ceil(len(values) * 0.70)))
                coarse_rmse = float(np.sqrt(np.mean(values[:keep])))
            else:
                coarse_rmse = math.inf
            overlap = count / max(1, len(current_coarse))
            # One extra loose correspondence must not outweigh a much cleaner
            # angular alignment. Penalize missing overlap without making it
            # the lexicographically dominant objective.
            robust_cost = (
                coarse_rmse
                + (1.0 - overlap) * coarse_limit * 0.35
            )
            score = (-robust_cost, overlap, -abs(yaw_deg))
            if score > best_score:
                best_score = score
                best_estimate = estimate
        return best_estimate

    def _identity_is_reliable(self, reference, current):
        _indices, squared = self._nearest(reference, current)
        accepted = squared <= self.max_correspondence_m ** 2
        count = int(np.count_nonzero(accepted))
        if count < self.min_points:
            return False
        ratio = float(count) / max(1, len(current))
        values = np.sort(squared[accepted])
        keep = max(self.min_points, int(math.ceil(len(values) * 0.70)))
        rmse = float(np.sqrt(np.mean(values[:keep])))
        return (
            ratio >= 0.98
            and rmse <= min(
                0.055,
                self.max_correspondence_m * 0.18,
            )
        )

    def _evaluate(self, reference, current, estimate):
        transformed = estimate.transform_points(current)
        _indices, squared = self._nearest(reference, transformed)
        accepted = squared <= self.max_correspondence_m ** 2
        count = int(np.count_nonzero(accepted))
        if count < self.min_points:
            return math.inf, count
        values = np.sort(squared[accepted])
        keep = max(self.min_points, int(math.ceil(len(values) * self.trim_fraction)))
        return float(np.sqrt(np.mean(values[:keep]))), count

    def evaluate(self, reference, current, estimate):
        """评估给定位姿的 RMSE、对应点数量和匹配比例。"""
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        rmse, count = self._evaluate(reference, current, estimate)
        return rmse, count, self._ratio(count, reference, current)

    @staticmethod
    def _ratio(count, reference, current):
        # 每个 current 点最多产生一个最近邻对应，因此分母必须是
        # current 点数，确保比例始终处于 0..1。
        denominator = max(1, len(current))
        return float(count) / denominator

    @staticmethod
    def _uniform_sample(points, maximum):
        if len(points) <= maximum:
            return points
        indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
        return points[indices]

    @staticmethod
    def _nearest(reference, query):
        indices = np.empty(len(query), dtype=np.int64)
        distances = np.empty(len(query), dtype=np.float64)
        for start in range(0, len(query), 128):
            chunk = query[start:start + 128]
            differences = chunk[:, None, :] - reference[None, :, :]
            squared = np.einsum("ijk,ijk->ij", differences, differences)
            nearest = np.argmin(squared, axis=1)
            rows = np.arange(len(chunk))
            indices[start:start + len(chunk)] = nearest
            distances[start:start + len(chunk)] = squared[rows, nearest]
        return indices, distances

    @staticmethod
    def _rigid_transform(source, target):
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        u_matrix, _values, vt_matrix = np.linalg.svd(covariance)
        rotation = vt_matrix.T @ u_matrix.T
        if np.linalg.det(rotation) < 0:
            vt_matrix[-1, :] *= -1
            rotation = vt_matrix.T @ u_matrix.T
        translation = target_center - rotation @ source_center
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return Pose2D(float(translation[0]), float(translation[1]), yaw)
