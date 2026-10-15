from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.inspection_annotation import (  # noqa: E402
    Qwen3VLInspectionBackend,
    discover_manifest,
    run_annotation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Qwen3-VL on RTX 3090 to annotate map-inspection photographs."
    )
    parser.add_argument(
        "--inspection",
        required=True,
        help="巡检批次目录、inspection_manifest.json，或包含多个批次的根目录（自动选择最新批次）",
    )
    parser.add_argument(
        "--model-path",
        default=str(ROOT / "models" / "qwen" / "Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-map", default=None, help="树莓派原始永久语义 JSON（只读）")
    parser.add_argument("--output", default=None, help="标注 JSON；默认写入所选巡检批次")
    parser.add_argument("--semantic-output", default=None, help="合并后的新语义 JSON")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-time-seconds", type=float, default=120.0)
    parser.add_argument("--min-visual-tokens", type=int, default=128)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个目标；0 表示全部")
    parser.add_argument("--force", action="store_true", help="忽略断点标注，全部重新推理")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许 transformers 联网找模型；正常离线部署不要开启",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")
    manifest = discover_manifest(args.inspection)
    output = Path(args.output).expanduser() if args.output else manifest.parent / "qwen_annotations.json"
    semantic_output = (
        Path(args.semantic_output).expanduser()
        if args.semantic_output
        else manifest.parent / "semantic_map_qwen.json"
    )
    semantic_source = Path(args.semantic_map).expanduser() if args.semantic_map else None
    print(f"巡检清单：{manifest}", flush=True)
    print(f"标注输出：{output}", flush=True)
    backend = Qwen3VLInspectionBackend(
        model_path=args.model_path,
        device=args.device,
        local_files_only=not args.allow_download,
        max_new_tokens=args.max_new_tokens,
        max_time_seconds=args.max_time_seconds,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
    )
    result = run_annotation(
        manifest_path=manifest,
        backend=backend,
        output_path=output,
        semantic_source=semantic_source,
        semantic_output=semantic_output,
        minimum_confidence=args.min_confidence,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
