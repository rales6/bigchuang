"""Precompute the CPU ORB reference database used by visual localization."""

from __future__ import annotations

import argparse
import json

from raspberry_pi.visual_place_localization import VisualPlaceRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build visual place reference features")
    parser.add_argument(
        "--references",
        default="inspections/ble_scene_17",
        help="inspection batch root or one inspection_manifest.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="feature directory; defaults to <manifest>/visual_features",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recognizer = VisualPlaceRecognizer(args.references)
    if not recognizer.available:
        raise SystemExit(
            "[VISUAL_LOCALIZATION] No valid reference images/manifest; "
            "nothing was changed."
        )
    try:
        index = recognizer.build_feature_database(args.output)
    except RuntimeError as exc:
        raise SystemExit(f"[VISUAL_LOCALIZATION] {exc}") from exc
    payload = json.loads(index.read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "ok",
        "database": str(index.resolve()),
        "algorithm": payload["algorithm"],
        "places": payload["place_count"],
        "images": payload["image_count"],
        "features": sum(item["feature_count"] for item in payload["images"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
