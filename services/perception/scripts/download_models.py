from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_qwen(source: str) -> None:
    target = ROOT / "models" / "qwen" / "Qwen3-VL-2B-Instruct"
    target.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(
            "Qwen/Qwen3-VL-2B-Instruct",
            local_dir=str(target),
        )
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Qwen/Qwen3-VL-2B-Instruct",
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    print(f"Qwen model saved to: {target}")


def download_yolo() -> None:
    from ultralytics import YOLO

    target_dir = ROOT / "models" / "yolo"
    target_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n-seg.pt")
    source = Path(getattr(model, "ckpt_path", "yolo11n-seg.pt"))
    if not source.exists():
        source = Path("yolo11n-seg.pt")
    target = target_dir / "yolo11n-seg.pt"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print(f"YOLO model saved to: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()

    if not args.qwen and not args.yolo:
        parser.error("Select --qwen and/or --yolo")
    if args.qwen:
        download_qwen(args.source)
    if args.yolo:
        download_yolo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
