"""Offline single-image/folder test for CPU visual place recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raspberry_pi.visual_place_localization import (
    VisualPlaceRecognizer,
    fuse_visual_top_k,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="offline visual localization test")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--folder")
    parser.add_argument("--references", default="inspections/ble_scene_17")
    parser.add_argument("--feature-database", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--min-margin", type=float, default=0.055)
    parser.add_argument("--min-inliers", type=int, default=9)
    parser.add_argument("--debug-directory", default=None)
    return parser


def _queries(args) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    folder = Path(args.folder)
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = build_parser().parse_args()
    recognizer = VisualPlaceRecognizer(
        args.references,
        minimum_score=args.min_score,
        minimum_margin=args.min_margin,
        minimum_inliers=args.min_inliers,
        feature_database=args.feature_database,
    )
    if not recognizer.available:
        raise SystemExit(
            "[VISUAL_LOCALIZATION] Reference database unavailable. "
            "Fallback would use global active localization."
        )
    queries = _queries(args)
    if not queries:
        raise SystemExit("[VISUAL_LOCALIZATION] No readable query images found.")
    history: dict[str, float] = {}
    output = []
    for index, query in enumerate(queries, start=1):
        try:
            matches = recognizer.rank([query], top_k=args.top_k)
        except RuntimeError as exc:
            raise SystemExit(f"[VISUAL_LOCALIZATION] {exc}") from exc
        fused, history = fuse_visual_top_k(
            history, matches, current_weight=0.60, top_k=args.top_k
        )
        top = matches[0] if matches else None
        record = {
            "query": str(query.resolve()),
            "extracted_result_count": len(matches),
            "top_k": [item.to_dict() for item in matches],
            "history_fused_top_k": fused,
            "reliable": bool(top is not None and top.reliable),
            "fallback_to_global_active_localization": bool(
                top is None or not top.reliable
            ),
        }
        if args.debug_directory and top is not None and top.reference_images:
            debug_path = Path(args.debug_directory) / f"{index:03d}_{query.stem}_top1.jpg"
            recognizer.save_debug_match(query, top.reference_images[0], debug_path)
            record["debug_match"] = str(debug_path.resolve())
        output.append(record)
    print(json.dumps({
        "reference_places": len(recognizer.places),
        "query_count": len(queries),
        "results": output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
