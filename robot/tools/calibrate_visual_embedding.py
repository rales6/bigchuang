"""Calibrate visual place embeddings using two real inspection batches.

Run this on the Raspberry Pi from the repository root.  It compares every
photo pair in the penultimate inspection batch with every reference place in
the latest batch.  The output is JSON so thresholds can be selected from real
same-place and different-place distributions instead of model confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from raspberry_pi.visual_place_localization import load_reference_places


def _post_similarity(
    endpoint: str,
    current: tuple[Path, Path],
    reference: tuple[Path, Path],
    timeout_s: float,
) -> float:
    boundary = "----visual-calibration-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, path in zip(
        ("current_left", "current_right", "reference_left", "reference_right"),
        (*current, *reference),
    ):
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: image/jpeg\r\n\r\n",
            path.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    request = urllib_request.Request(
        endpoint.rstrip("/") + "/localize/embedding",
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"visual embedding request failed: {exc}") from exc
    if not isinstance(value, dict) or not value.get("ok"):
        raise RuntimeError(str(value))
    return float(value["similarity"])


def _capture_images(manifest: Path, record: dict) -> tuple[Path, Path] | None:
    images: list[Path] = []
    for key in ("image", "camera_2_image"):
        name = record.get(key)
        if not name:
            continue
        path = Path(str(name))
        if not path.is_absolute():
            path = manifest.parent / path
        if path.is_file():
            images.append(path.resolve())
    if not images:
        return None
    return images[0], images[1] if len(images) > 1 else images[0]


def run(args: argparse.Namespace) -> dict:
    root = Path(args.reference_root)
    manifests = sorted(root.rglob("inspection_manifest.json"), key=lambda path: path.stat().st_mtime_ns)
    if len(manifests) < 2:
        raise RuntimeError("at least two inspection manifests are required")
    query_manifest, reference_manifest = manifests[-2], manifests[-1]
    references = load_reference_places(reference_manifest)
    query_payload = json.loads(query_manifest.read_text(encoding="utf-8"))
    queries = []
    for record in query_payload.get("captures") or []:
        images = _capture_images(query_manifest, record)
        if images is not None:
            queries.append((str(record.get("target_id") or ""), images))
    if args.maximum_queries > 0:
        queries = queries[: args.maximum_queries]

    rows = []
    same_scores: list[float] = []
    wrong_scores: list[float] = []
    correct = 0
    for offset, (target_id, query_images) in enumerate(queries, 1):
        ranked = []
        for place in references:
            reference_images = (
                place.images[0],
                place.images[1] if len(place.images) > 1 else place.images[0],
            )
            score = _post_similarity(args.endpoint, query_images, reference_images, args.timeout)
            ranked.append((score, place.target_id))
        ranked.sort(reverse=True)
        best_score, best_id = ranked[0]
        same = next((score for score, item_id in ranked if item_id == target_id), None)
        best_wrong = next((score for score, item_id in ranked if item_id != target_id), None)
        if same is not None:
            same_scores.append(float(same))
        if best_wrong is not None:
            wrong_scores.append(float(best_wrong))
        if best_id == target_id:
            correct += 1
        row = {
            "query": target_id,
            "best": best_id,
            "best_score": round(best_score, 6),
            "same_score": None if same is None else round(float(same), 6),
            "best_wrong_score": None if best_wrong is None else round(float(best_wrong), 6),
            "margin": None if len(ranked) < 2 else round(ranked[0][0] - ranked[1][0], 6),
            "top3": [{"target_id": item_id, "score": round(score, 6)} for score, item_id in ranked[:3]],
        }
        rows.append(row)
        print(json.dumps({"progress": f"{offset}/{len(queries)}", **row}, ensure_ascii=False), flush=True)

    return {
        "query_manifest": str(query_manifest),
        "reference_manifest": str(reference_manifest),
        "queries": len(rows),
        "top1_correct": correct,
        "top1_accuracy": 0.0 if not rows else round(correct / len(rows), 4),
        "same_min": None if not same_scores else round(min(same_scores), 6),
        "same_median": None if not same_scores else round(float(np.median(same_scores)), 6),
        "wrong_max": None if not wrong_scores else round(max(wrong_scores), 6),
        "wrong_median": None if not wrong_scores else round(float(np.median(wrong_scores)), 6),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", default="inspections/ble_scene_17")
    parser.add_argument("--endpoint", default="http://192.0.2.20:8011")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--maximum-queries", type=int, default=0)
    parser.add_argument("--output", default="search_runs/visual_embedding_calibration.json")
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
