from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.grounding.qwen_grounder import Qwen3VLTargetGrounder


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Qwen3-VL target grounding on one image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--command", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    grounder = Qwen3VLTargetGrounder(
        model_path=str(model_path),
        device=args.device,
        local_files_only=True,
        max_new_tokens=args.max_new_tokens,
    )
    result = grounder.ground(frame, args.command)
    print(result)

    if result.found and result.bbox is not None:
        box = result.bbox
        cv2.rectangle(
            frame,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            (0, 255, 0),
            2,
        )
        output_path = ROOT / "outputs" / "qwen_grounding_result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
