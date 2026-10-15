"""Cross-batch SIFT/RANSAC calibration for permanent inspection photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _image_paths(manifest: Path, record: dict) -> tuple[Path, ...]:
    result = []
    for key in ("image", "camera_2_image"):
        name = record.get(key)
        if name:
            path = manifest.parent / str(name)
            if path.is_file():
                result.append(path)
    return tuple(result)


def _features(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [], None
    maximum = max(image.shape)
    if maximum > 960:
        scale = 960.0 / maximum
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    detector = cv2.SIFT_create(nfeatures=2200, contrastThreshold=0.025, edgeThreshold=12)
    return detector.detectAndCompute(image, None)


def _compare(first, second) -> dict:
    first_keypoints, first_descriptors = first
    second_keypoints, second_descriptors = second
    if first_descriptors is None or second_descriptors is None:
        return {"score": 0.0, "inliers": 0, "good": 0, "coverage": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    good = [one for one, two in pairs if one.distance < 0.72 * two.distance]
    inliers = 0
    coverage = 0.0
    if len(good) >= 6:
        source = np.float32([first_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([second_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 4.0)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            if inliers >= 3:
                points = source.reshape(-1, 2)[inlier_mask]
                span = np.ptp(points, axis=0)
                coverage = float(min(1.0, (span[0] * span[1]) / (960.0 * 720.0 * 0.20)))
    ratio = inliers / max(1, len(good))
    score = (
        0.46 * min(1.0, inliers / 30.0)
        + 0.20 * min(1.0, len(good) / 70.0)
        + 0.22 * ratio
        + 0.12 * coverage
    )
    return {"score": float(score), "inliers": inliers, "good": len(good), "coverage": coverage}


def _place_score(query_features, reference_features) -> dict:
    alternatives = []
    direct = [_compare(query_features[i], reference_features[i]) for i in range(min(len(query_features), len(reference_features)))]
    alternatives.extend(direct)
    if len(query_features) >= 2 and len(reference_features) >= 2:
        alternatives.extend([
            _compare(query_features[0], reference_features[1]),
            _compare(query_features[1], reference_features[0]),
        ])
    alternatives.sort(key=lambda item: item["score"], reverse=True)
    best = dict(alternatives[0]) if alternatives else {"score": 0.0, "inliers": 0, "good": 0, "coverage": 0.0}
    if len(alternatives) > 1:
        best["score"] += 0.20 * alternatives[1]["score"]
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", default="inspections/ble_scene_17")
    parser.add_argument("--output", default="search_runs/sift_visual_calibration.json")
    args = parser.parse_args()
    manifests = sorted(Path(args.reference_root).rglob("inspection_manifest.json"), key=lambda path: path.stat().st_mtime_ns)
    if len(manifests) < 2:
        raise RuntimeError("at least two inspection batches are required")
    previous, latest = manifests[-2:]
    previous_records = json.loads(previous.read_text(encoding="utf-8")).get("captures") or []
    latest_records = json.loads(latest.read_text(encoding="utf-8")).get("captures") or []
    references = []
    for record in latest_records:
        paths = _image_paths(latest, record)
        if paths:
            references.append((str(record.get("target_id") or ""), [_features(path) for path in paths]))
    rows = []
    correct = 0
    for record in previous_records:
        target_id = str(record.get("target_id") or "")
        paths = _image_paths(previous, record)
        query = [_features(path) for path in paths]
        ranked = []
        for reference_id, reference_features in references:
            result = _place_score(query, reference_features)
            ranked.append((result["score"], reference_id, result))
        ranked.sort(reverse=True)
        if ranked and ranked[0][1] == target_id:
            correct += 1
        same = next((item for item in ranked if item[1] == target_id), None)
        row = {
            "query": target_id,
            "best": None if not ranked else ranked[0][1],
            "same_rank": None if same is None else next(index for index, item in enumerate(ranked, 1) if item[1] == target_id),
            "same_score": None if same is None else round(same[0], 6),
            "margin": None if len(ranked) < 2 else round(ranked[0][0] - ranked[1][0], 6),
            "top3": [
                {"target_id": item_id, **{key: round(value, 6) if isinstance(value, float) else value for key, value in result.items()}}
                for _, item_id, result in ranked[:3]
            ],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    output = {
        "queries": len(rows),
        "top1_correct": correct,
        "top1_accuracy": round(correct / max(1, len(rows)), 4),
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
