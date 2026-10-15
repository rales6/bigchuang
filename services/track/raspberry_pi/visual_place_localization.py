"""Camera place recognition against permanent inspection photographs.

The module never changes the permanent map.  It is only a second opinion for
startup localization when different parts of the occupancy map have similar
geometry.  OpenCV is imported lazily because simulation/unit-test computers do
not need the Raspberry Pi camera dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from raspberry_pi.mapping.geometry import Pose2D


@dataclass(frozen=True)
class ReferencePlace:
    target_id: str
    pose: Pose2D
    images: tuple[Path, ...]
    manifest: Path


@dataclass(frozen=True)
class PermanentLandmark:
    target_id: str
    x_m: float
    y_m: float
    labels: frozenset[str]


@dataclass(frozen=True)
class LandmarkCandidateScore:
    candidate_index: int
    score: float
    matched_labels: tuple[str, ...]
    expected_labels: tuple[str, ...]


_DISTINCTIVE_LABEL_WEIGHTS = {
    "cabinet": 3.0,
    "chair": 2.5,
    "table": 2.5,
    "door": 3.0,
    "door_frame": 2.5,
    "wood_grain": 1.5,
    "glass": 2.0,
    "column": 2.5,
    "fixed_equipment": 3.0,
}


def _landmark_labels(*values: Any) -> frozenset[str]:
    text = " ".join(str(value or "").lower().replace("-", "_") for value in values)
    labels = set()
    aliases = {
        "cabinet": ("cabinet", "cupboard", "柜", "橱"),
        "chair": ("chair", "椅"),
        "table": ("table", "desk", "桌"),
        "door_frame": ("door_frame", "doorframe", "门框"),
        "door": (" door ", "门"),
        "wood_grain": ("wood_grain", "woodgrain", "木纹"),
        "glass": ("glass", "玻璃"),
        "column": ("column", "pillar", "柱"),
        "fixed_equipment": ("fixed_equipment", "固定设备"),
    }
    padded = f" {text} "
    for label, words in aliases.items():
        if any(word in padded for word in words):
            labels.add(label)
    # A door frame is a stronger, more specific landmark than a generic door.
    if "door_frame" in labels:
        labels.discard("door")
    return frozenset(labels)


def load_permanent_landmarks(semantic_map: str | Path | None) -> list[PermanentLandmark]:
    if semantic_map is None:
        return []
    path = Path(semantic_map)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    result = []
    entities = payload.get("entities") if isinstance(payload, dict) else {}
    for target_id, entity in (entities or {}).items():
        if not isinstance(entity, dict) or entity.get("permanent") is False:
            continue
        annotation = entity.get("qwen_annotation")
        annotation = annotation if isinstance(annotation, dict) else {}
        attributes = annotation.get("visible_attributes")
        labels = _landmark_labels(
            entity.get("semantic_label"),
            annotation.get("name_zh"),
            annotation.get("category"),
            annotation.get("description_zh"),
            " ".join(str(item) for item in attributes) if isinstance(attributes, list) else "",
        )
        if not labels:
            continue
        try:
            x_m = float(entity.get("target_x_m"))
            y_m = float(entity.get("target_y_m"))
        except (TypeError, ValueError):
            continue
        result.append(PermanentLandmark(str(target_id), x_m, y_m, labels))
    return result


def detection_landmark_labels(detections: list[dict[str, Any]]) -> frozenset[str]:
    labels = set()
    for detection in detections:
        if not isinstance(detection, dict) or float(detection.get("confidence") or 0.0) < 0.48:
            continue
        attributes = detection.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        labels.update(_landmark_labels(
            detection.get("class_name"),
            detection.get("appearance_key"),
            attributes.get("description"),
            attributes.get("material"),
        ))
    return frozenset(labels)


def rank_landmark_candidates(
    candidates: list[Any],
    landmarks: list[PermanentLandmark],
    detected_labels: frozenset[str],
    *,
    maximum_distance_m: float = 3.5,
    field_of_view_deg: float = 150.0,
    known_map: Any | None = None,
) -> list[LandmarkCandidateScore]:
    """Rank radar basins by permanent landmarks expected in the camera view."""
    if not candidates or not landmarks or not detected_labels:
        return []
    half_fov = math.radians(float(field_of_view_deg)) / 2.0
    ranked = []
    for index, candidate in enumerate(candidates):
        pose = candidate.pose if hasattr(candidate, "pose") else candidate
        expected = set()
        for landmark in landmarks:
            dx = landmark.x_m - pose.x_m
            dy = landmark.y_m - pose.y_m
            distance = math.hypot(dx, dy)
            bearing = math.atan2(
                math.sin(math.atan2(dy, dx) - pose.yaw_rad),
                math.cos(math.atan2(dy, dx) - pose.yaw_rad),
            )
            visible = distance <= float(maximum_distance_m) and abs(bearing) <= half_fov
            if visible and known_map is not None and distance > 0.30:
                # The semantic target itself is usually an occupied map cell.
                # End the visibility ray just before it, while still rejecting
                # any intervening table, cabinet or wall.
                cutback = min(0.60, max(0.35, distance * 0.16))
                end_x = landmark.x_m - dx / distance * cutback
                end_y = landmark.y_m - dy / distance * cutback
                try:
                    try:
                        visible = bool(known_map.line_is_clear(
                            (pose.x_m, pose.y_m),
                            (end_x, end_y),
                            mask=getattr(known_map, "free", None),
                        ))
                    except TypeError:
                        visible = bool(known_map.line_is_clear(
                            (pose.x_m, pose.y_m), (end_x, end_y)
                        ))
                except (AttributeError, TypeError, ValueError):
                    visible = False
            if visible:
                expected.update(landmark.labels)
        matched = detected_labels & expected
        score = sum(_DISTINCTIVE_LABEL_WEIGHTS.get(label, 1.0) for label in matched)
        # Seeing several independent fixed types is substantially stronger
        # than repeatedly seeing the same generic object class.
        if len(matched) >= 2:
            score += 0.75 * (len(matched) - 1)
        ranked.append(LandmarkCandidateScore(
            candidate_index=index,
            score=float(score),
            matched_labels=tuple(sorted(matched)),
            expected_labels=tuple(sorted(expected)),
        ))
    return sorted(ranked, key=lambda item: (item.score, -item.candidate_index), reverse=True)


@dataclass(frozen=True)
class VisualPlaceMatch:
    target_id: str
    pose: Pose2D
    score: float
    runner_up_score: float
    margin: float
    inliers: int
    good_matches: int
    reference_images: tuple[str, ...]
    manifest: str
    reliable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "pose": [self.pose.x_m, self.pose.y_m, self.pose.yaw_rad],
            "score": round(self.score, 4),
            "runner_up_score": round(self.runner_up_score, 4),
            "margin": round(self.margin, 4),
            "inliers": self.inliers,
            "good_matches": self.good_matches,
            "reference_images": list(self.reference_images),
            "manifest": self.manifest,
            "reliable": self.reliable,
        }


def latest_manifest(reference_root: str | Path) -> Path | None:
    root = Path(reference_root)
    if root.is_file():
        return root if root.name == "inspection_manifest.json" else None
    manifests = list(root.rglob("inspection_manifest.json")) if root.exists() else []
    return max(manifests, key=lambda path: path.stat().st_mtime_ns) if manifests else None


def load_reference_places(reference_root: str | Path) -> list[ReferencePlace]:
    manifest = latest_manifest(reference_root)
    if manifest is None:
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    places: list[ReferencePlace] = []
    for index, record in enumerate(payload.get("captures") or []):
        if not isinstance(record, dict):
            continue
        try:
            pose = Pose2D(
                float(record["x_m"]),
                float(record["y_m"]),
                math.radians(float(record["yaw_deg"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
        names = [record.get("image"), record.get("camera_2_image")]
        images = []
        for name in names:
            if not name:
                continue
            path = Path(str(name))
            if not path.is_absolute():
                path = manifest.parent / path
            if path.is_file():
                images.append(path.resolve())
        if images:
            places.append(ReferencePlace(
                target_id=str(record.get("target_id") or f"reference_{index + 1:02d}"),
                pose=pose,
                images=tuple(images),
                manifest=manifest.resolve(),
            ))
    return places


def select_visual_match(
    ranked: list[tuple[float, int, int, ReferencePlace]],
    *,
    minimum_score: float,
    minimum_margin: float,
    minimum_inliers: int,
) -> VisualPlaceMatch | None:
    if not ranked:
        return None
    ranked = sorted(ranked, key=lambda item: (item[0], item[1], item[2]), reverse=True)
    score, inliers, good_matches, place = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = float(score - runner_up)
    reliable = bool(
        score >= float(minimum_score)
        and margin >= float(minimum_margin)
        and inliers >= int(minimum_inliers)
    )
    return VisualPlaceMatch(
        target_id=place.target_id,
        pose=place.pose,
        score=float(score),
        runner_up_score=float(runner_up),
        margin=margin,
        inliers=int(inliers),
        good_matches=int(good_matches),
        reference_images=tuple(str(path) for path in place.images),
        manifest=str(place.manifest),
        reliable=reliable,
    )


class VisualPlaceRecognizer:
    """ORB/RANSAC place matcher using the latest complete inspection batch."""

    def __init__(
        self,
        reference_root: str | Path,
        *,
        semantic_map: str | Path | None = None,
        minimum_score: float = 0.30,
        minimum_margin: float = 0.055,
        minimum_inliers: int = 9,
    ) -> None:
        self.reference_root = Path(reference_root)
        self.minimum_score = float(minimum_score)
        self.minimum_margin = float(minimum_margin)
        self.minimum_inliers = int(minimum_inliers)
        self.places = load_reference_places(self.reference_root)
        self.landmarks = load_permanent_landmarks(semantic_map)
        self._cache: dict[Path, tuple[Any, Any, Any]] = {}

    @property
    def available(self) -> bool:
        return bool(self.places)

    @staticmethod
    def _opencv():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("视觉定位需要树莓派安装 python3-opencv") from exc
        return cv2

    def _features(self, path: Path):
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        cv2 = self._opencv()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            result = ([], None, None)
            self._cache[path] = result
            return result
        maximum = max(image.shape[:2])
        if maximum > 960:
            scale = 960.0 / maximum
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        detector = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8, fastThreshold=12)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        histogram = cv2.calcHist([image], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)
        result = (keypoints or [], descriptors, histogram)
        self._cache[path] = result
        return result

    def _compare(self, current: Path, reference: Path) -> tuple[float, int, int]:
        cv2 = self._opencv()
        current_keypoints, current_desc, current_hist = self._features(current)
        reference_keypoints, reference_desc, reference_hist = self._features(reference)
        if current_desc is None or reference_desc is None:
            return 0.0, 0, 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(current_desc, reference_desc, k=2)
        good = [first for first, second in pairs if first.distance < 0.76 * second.distance]
        inliers = 0
        if len(good) >= 6:
            import numpy as np

            source = np.float32([current_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
            target = np.float32([reference_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
            _matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.ravel().sum())
        histogram_score = max(0.0, float(cv2.compareHist(current_hist, reference_hist, cv2.HISTCMP_CORREL)))
        score = (
            0.68 * min(1.0, inliers / 24.0)
            + 0.22 * min(1.0, len(good) / 55.0)
            + 0.10 * histogram_score
        )
        return float(score), inliers, len(good)

    def match(self, current_images: list[str | Path]) -> VisualPlaceMatch | None:
        current = [Path(path) for path in current_images if Path(path).is_file()]
        if not current or not self.places:
            return None
        ranked: list[tuple[float, int, int, ReferencePlace]] = []
        for place in self.places:
            direct = [
                self._compare(current[index], place.images[index])
                for index in range(min(len(current), len(place.images)))
            ]
            cross = []
            if len(current) >= 2 and len(place.images) >= 2:
                cross = [
                    self._compare(current[0], place.images[1]),
                    self._compare(current[1], place.images[0]),
                ]

            def combine(items, penalty=1.0):
                if not items:
                    return 0.0, 0, 0
                scores = sorted((item[0] for item in items), reverse=True)
                score = (scores[0] + (0.30 * scores[1] if len(scores) > 1 else 0.0)) * penalty
                return score, sum(item[1] for item in items), sum(item[2] for item in items)

            best = max(combine(direct), combine(cross, 0.94), key=lambda item: item[0])
            ranked.append((best[0], best[1], best[2], place))
        return select_visual_match(
            ranked,
            minimum_score=self.minimum_score,
            minimum_margin=self.minimum_margin,
            minimum_inliers=self.minimum_inliers,
        )

    def build_reference_sheets(
        self,
        output_directory: str | Path,
        *,
        columns: int = 4,
        tile_size: tuple[int, int] = (300, 220),
    ) -> tuple[Path, Path, list[dict[str, Any]]]:
        """Build Qwen-readable paired contact sheets with stable Rxx labels."""
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        columns = max(1, int(columns))
        width, height = tile_size
        label_height = 28
        rows = max(1, int(math.ceil(len(self.places) / columns)))
        sheets = [
            Image.new("RGB", (columns * width, rows * height), (28, 28, 28))
            for _ in range(2)
        ]
        references = []
        for offset, place in enumerate(self.places):
            index = offset + 1
            row, column = divmod(offset, columns)
            left, top = column * width, row * height
            references.append({
                "index": index,
                "target_id": place.target_id,
                "x_m": round(place.pose.x_m, 4),
                "y_m": round(place.pose.y_m, 4),
                "yaw_deg": round(math.degrees(place.pose.yaw_rad), 2),
            })
            for camera_index, sheet in enumerate(sheets):
                image_path = (
                    place.images[camera_index]
                    if camera_index < len(place.images)
                    else place.images[0]
                )
                try:
                    tile = Image.open(image_path).convert("RGB")
                    tile.thumbnail((width, height - label_height))
                    canvas = Image.new("RGB", (width, height - label_height), (8, 8, 8))
                    canvas.paste(
                        tile,
                        ((canvas.width - tile.width) // 2, (canvas.height - tile.height) // 2),
                    )
                    sheet.paste(canvas, (left, top + label_height))
                except OSError:
                    pass
                draw = ImageDraw.Draw(sheet)
                draw.rectangle((left, top, left + width - 1, top + label_height - 1), fill=(255, 220, 40))
                draw.text(
                    (left + 7, top + 5),
                    f"R{index:02d}  {place.target_id}",
                    fill=(0, 0, 0),
                    stroke_width=1,
                )
                draw.rectangle((left, top, left + width - 1, top + height - 1), outline=(255, 255, 255), width=2)
        left_path = output / "visual_reference_left.jpg"
        right_path = output / "visual_reference_right.jpg"
        sheets[0].save(left_path, quality=88)
        sheets[1].save(right_path, quality=88)
        return left_path, right_path, references

    def place_for_index(self, index: int) -> ReferencePlace | None:
        offset = int(index) - 1
        return self.places[offset] if 0 <= offset < len(self.places) else None

    def places_for_radar_candidates(
        self,
        candidates: list[Any],
        *,
        maximum: int = 4,
    ) -> list[ReferencePlace]:
        """Pick one photographed viewpoint nearest each distinct radar basin."""
        selected: list[ReferencePlace] = []
        for candidate in candidates:
            pose = candidate.pose if hasattr(candidate, "pose") else candidate
            choices = sorted(
                self.places,
                key=lambda place: (
                    math.hypot(place.pose.x_m - pose.x_m, place.pose.y_m - pose.y_m)
                    + 0.20 * abs(math.sin((place.pose.yaw_rad - pose.yaw_rad) / 2.0))
                ),
            )
            place = next((item for item in choices if item not in selected), None)
            if place is not None:
                selected.append(place)
            if len(selected) >= max(1, int(maximum)):
                break
        return selected
