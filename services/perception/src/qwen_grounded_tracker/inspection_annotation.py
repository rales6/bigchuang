from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image, ImageDraw


ANNOTATION_SCHEMA = "inspection_semantic_annotations.v1"
FIXED_STATES = {"fixed", "probably_fixed"}
UNKNOWN_LABELS = {"", "unknown", "unclassified", "uncertain"}


class InspectionVisionBackend(Protocol):
    def annotate(self, image_paths: Sequence[Path], prompt: str) -> str: ...


def extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen output does not contain a JSON object")
    candidate = cleaned[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        value = json.loads(repaired)
    if not isinstance(value, dict):
        raise ValueError("Qwen output root must be a JSON object")
    return value


def discover_manifest(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != "inspection_manifest.json":
            raise ValueError(f"Expected inspection_manifest.json, got: {candidate}")
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    direct = candidate / "inspection_manifest.json"
    if direct.exists():
        return direct
    manifests = sorted(
        candidate.rglob("inspection_manifest.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No inspection_manifest.json found under {candidate}")
    return manifests[0]


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("captures"), list):
        raise ValueError(f"Invalid inspection manifest: {path}")
    return payload


def resolve_record_images(record: dict[str, Any], directory: Path) -> list[Path]:
    values: list[str] = []
    camera_images = record.get("camera_images")
    if isinstance(camera_images, list):
        values.extend(str(item) for item in camera_images if item)
    for key in ("image", "camera_2_image"):
        value = record.get(key)
        if value:
            values.append(str(value))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        item = Path(value).expanduser()
        if not item.is_absolute():
            item = directory / item
        item = item.resolve()
        if item in seen:
            continue
        if not item.is_file():
            raise FileNotFoundError(f"Inspection image is missing: {item}")
        seen.add(item)
        resolved.append(item)
    if not resolved:
        raise ValueError(f"Capture {record.get('target_id', '?')} has no images")
    return resolved


def _clamp_number(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def _normalise_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower().replace("-", " ")
    label = re.sub(r"[^a-z0-9_ ]+", "", label)
    label = re.sub(r"[ _]+", "_", label).strip("_")
    return label or "unknown"


def _normalise_views(raw_views: Any, image_count: int) -> list[dict[str, Any]]:
    if not isinstance(raw_views, list):
        return []
    result: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in raw_views:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("image_index"))
        except (TypeError, ValueError):
            continue
        if index < 1 or index > image_count or index in used:
            continue
        visible = bool(item.get("target_visible", False))
        bbox = item.get("bbox_2d")
        parsed_bbox: list[float] = []
        if visible and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                parsed_bbox = [
                    _clamp_number(bbox[position], 0.0, 1000.0, 0.0)
                    for position in range(4)
                ]
            except (TypeError, ValueError):
                parsed_bbox = []
            if len(parsed_bbox) == 4:
                x1, y1, x2, y2 = parsed_bbox
                if x2 <= x1 or y2 <= y1:
                    parsed_bbox = []
        result.append(
            {
                "image_index": index,
                "target_visible": visible and bool(parsed_bbox),
                "bbox_2d": parsed_bbox,
                "confidence": _clamp_number(item.get("confidence"), 0.0, 1.0, 0.0),
            }
        )
        used.add(index)
    return sorted(result, key=lambda item: item["image_index"])


def normalise_annotation(
    payload: dict[str, Any],
    record: dict[str, Any],
    images: Sequence[Path],
    minimum_confidence: float,
) -> dict[str, Any]:
    label = _normalise_label(payload.get("semantic_label"))
    confidence = _clamp_number(payload.get("confidence"), 0.0, 1.0, 0.0)
    fixed_state = str(payload.get("fixed_or_temporary") or "unknown").strip().lower()
    if fixed_state not in {"fixed", "probably_fixed", "temporary", "unknown"}:
        fixed_state = "unknown"
    views = _normalise_views(payload.get("views"), len(images))
    visible_count = sum(bool(item["target_visible"]) for item in views)
    model_review = bool(payload.get("needs_human_review", False))
    needs_review = (
        model_review
        or confidence < minimum_confidence
        or label in UNKNOWN_LABELS
        or visible_count == 0
        or fixed_state not in FIXED_STATES
    )
    attributes = payload.get("visible_attributes")
    if not isinstance(attributes, list):
        attributes = []
    return {
        "target_id": str(record.get("target_id") or "unknown_target"),
        "kind": str(record.get("kind") or "unknown"),
        "target_x_m": record.get("target_x_m"),
        "target_y_m": record.get("target_y_m"),
        "semantic_label": label,
        "name_zh": str(payload.get("name_zh") or label),
        "category": _normalise_label(payload.get("category")),
        "description_zh": str(payload.get("description_zh") or "").strip(),
        "visible_attributes": [str(item).strip() for item in attributes if str(item).strip()],
        "condition": str(payload.get("condition") or "unknown").strip(),
        "fixed_or_temporary": fixed_state,
        "same_entity_across_views": payload.get("same_entity_across_views"),
        "confidence": confidence,
        "evidence_quality": str(payload.get("evidence_quality") or "unknown").strip(),
        "needs_human_review": needs_review,
        "model_needs_human_review": model_review,
        "review_reason": str(payload.get("review_reason") or "").strip(),
        "views": views,
        "source_images": [path.name for path in images],
        "promoted_to_semantic_label": not needs_review,
    }


def build_prompt(record: dict[str, Any], image_count: int) -> str:
    subject = str(record.get("inspection_subject") or record.get("kind") or "unknown")
    question = str(record.get("review_question") or "识别这个地图目标的真实类别和可见特征")
    return f"""
你是室内移动机器人地图的离线语义标注器。现在有 {image_count} 张按顺序编号的巡检照片，
它们由机器人在同一个地图目标附近停车后拍摄，可能来自两路方向不同的摄像头。

地图记录：
- target_id: {record.get('target_id', 'unknown')}
- 几何类型: {record.get('kind', 'unknown')}
- 预期检查对象: {subject}
- 检查问题: {question}

任务：只标注这个 target_id 对应的墙面、开口或黑色障碍区域。不要把远处或前景中无关物体
误当成目标。结合全部照片判断它最可能是什么，例如 wall、glass_wall、door、window、table、
desk、cabinet、shelf、pillar、chair、box、fixed_obstacle。看不清就用 unknown，禁止猜测被遮挡的
背面、精确尺寸或照片中不存在的信息。semantic_label 和 category 必须是简短英文 snake_case；
中文说明写入 name_zh 和 description_zh。bbox 使用各自整张图片的 0-1000 相对坐标。

fixed_or_temporary 只能是 fixed、probably_fixed、temporary、unknown。只有墙体、门窗、桌柜等
稳定设施才可标为 fixed/probably_fixed。若证据不足、两个视角冲突或目标不可见，
needs_human_review 必须为 true。

只返回一个 JSON 对象，不要 Markdown：
{{
  "semantic_label":"table",
  "name_zh":"桌子",
  "category":"furniture",
  "description_zh":"可见的简短客观描述",
  "visible_attributes":["黑色桌腿","矩形桌面"],
  "condition":"normal",
  "fixed_or_temporary":"probably_fixed",
  "same_entity_across_views":true,
  "confidence":0.82,
  "evidence_quality":"good",
  "needs_human_review":false,
  "review_reason":"",
  "views":[
    {{"image_index":1,"target_visible":true,"bbox_2d":[120,180,820,900],"confidence":0.85}}
  ]
}}
views 必须逐张给出 image_index=1..{image_count}；不可见时 target_visible=false 且 bbox_2d=[]。
""".strip()


class Qwen3VLInspectionBackend:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        local_files_only: bool = True,
        max_new_tokens: int = 512,
        max_time_seconds: float = 120.0,
        min_visual_tokens: int = 128,
        max_visual_tokens: int = 768,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.device = device
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.max_time_seconds = max_time_seconds
        self.min_visual_tokens = min_visual_tokens
        self.max_visual_tokens = max_visual_tokens
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        model_path = Path(self.model_path)
        if self.local_files_only and not model_path.exists():
            raise FileNotFoundError(f"Qwen model not found: {model_path.resolve()}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        use_cuda = self.device.lower().startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        kwargs = {"local_files_only": self.local_files_only, "low_cpu_mem_usage": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, torch_dtype=dtype, **kwargs
            )
        self.model.to("cuda" if use_cuda else "cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size = {
                "longest_edge": int(self.max_visual_tokens * 32 * 32),
                "shortest_edge": int(self.min_visual_tokens * 32 * 32),
            }
        self.torch = torch
        print(f"Qwen3-VL 已加载：{self.model_path}，设备={self.device}", flush=True)

    def annotate(self, image_paths: Sequence[Path], prompt: str) -> str:
        self._ensure_loaded()
        assert self.model is not None and self.processor is not None and self.torch is not None
        content: list[dict[str, Any]] = []
        opened: list[Image.Image] = []
        try:
            for index, path in enumerate(image_paths, start=1):
                image = Image.open(path).convert("RGB")
                opened.append(image)
                content.extend(
                    [
                        {"type": "text", "text": f"巡检照片 image_index={index}:"},
                        {"type": "image", "image": image},
                    ]
                )
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    max_time=self.max_time_seconds,
                    do_sample=False,
                )
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            for image in opened:
                image.close()


def input_fingerprint(record: dict[str, Any], images: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    relevant = {
        key: record.get(key)
        for key in (
            "target_id",
            "kind",
            "target_x_m",
            "target_y_m",
            "inspection_subject",
            "review_question",
        )
    }
    digest.update(json.dumps(relevant, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for path in images:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def render_overlays(
    annotation: dict[str, Any],
    images: Sequence[Path],
    output_directory: Path,
) -> list[str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    by_index = {item["image_index"]: item for item in annotation.get("views", [])}
    outputs: list[str] = []
    for index, source in enumerate(images, start=1):
        target = output_directory / f"{annotation['target_id']}_camera_{index}.jpg"
        with Image.open(source).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            view = by_index.get(index)
            if view and view.get("target_visible") and len(view.get("bbox_2d", [])) == 4:
                width, height = image.size
                x1, y1, x2, y2 = view["bbox_2d"]
                box = (
                    int(x1 * width / 1000.0),
                    int(y1 * height / 1000.0),
                    int(x2 * width / 1000.0),
                    int(y2 * height / 1000.0),
                )
                line_width = max(2, int(min(width, height) / 240))
                draw.rectangle(box, outline=(255, 40, 40), width=line_width)
                draw.text((box[0] + 4, max(2, box[1] - 14)), annotation["semantic_label"], fill=(255, 40, 40))
            image.save(target, quality=92)
        outputs.append(str(target.resolve()))
    return outputs


def write_annotation_csv(path: Path, annotations: Sequence[dict[str, Any]]) -> None:
    fields = (
        "target_id",
        "kind",
        "target_x_m",
        "target_y_m",
        "semantic_label",
        "name_zh",
        "category",
        "description_zh",
        "condition",
        "fixed_or_temporary",
        "confidence",
        "needs_human_review",
        "promoted_to_semantic_label",
        "source_images",
        "annotated_images",
        "review_reason",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for annotation in annotations:
            row = {key: annotation.get(key, "") for key in fields}
            row["source_images"] = ";".join(annotation.get("source_images", []))
            row["annotated_images"] = ";".join(annotation.get("annotated_images", []))
            writer.writerow(row)


def merge_semantic_map(
    source_path: Path | None,
    output_path: Path,
    manifest: dict[str, Any],
    annotations: Sequence[dict[str, Any]],
) -> None:
    if source_path is not None and source_path.exists():
        source_resolved = source_path.resolve()
        if source_resolved == output_path.resolve():
            raise ValueError("Semantic output must not overwrite the source semantic map")
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid semantic map: {source_path}")
        merged = deepcopy(data)
    else:
        merged = {
            "format_version": 1,
            "map": manifest.get("map"),
            "entities": {},
        }
    entities = merged.setdefault("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("Semantic map entities must be an object")
    for annotation in annotations:
        target_id = annotation["target_id"]
        entity = entities.setdefault(target_id, {})
        entity.update(
            {
                "target_id": target_id,
                "kind": annotation["kind"],
                "target_x_m": annotation.get("target_x_m"),
                "target_y_m": annotation.get("target_y_m"),
                "permanent": True,
                "qwen_annotation": {
                    key: value
                    for key, value in annotation.items()
                    if key not in {"raw_response", "input_fingerprint"}
                },
            }
        )
        if annotation.get("promoted_to_semantic_label"):
            entity["semantic_label"] = annotation["semantic_label"]
            entity["semantic_label_source"] = "qwen3_vl_inspection"
            entity["semantic_label_confidence"] = annotation["confidence"]
        else:
            entity["qwen_candidate_label"] = annotation["semantic_label"]
            entity["qwen_candidate_confidence"] = annotation["confidence"]
    merged["qwen_annotation_schema"] = ANNOTATION_SCHEMA
    merged["qwen_annotation_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)


def run_annotation(
    manifest_path: Path,
    backend: InspectionVisionBackend,
    output_path: Path,
    semantic_source: Path | None = None,
    semantic_output: Path | None = None,
    minimum_confidence: float = 0.55,
    force: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    directory = manifest_path.parent
    previous: dict[str, dict[str, Any]] = {}
    if output_path.exists() and not force:
        old = json.loads(output_path.read_text(encoding="utf-8"))
        previous = {
            str(item.get("target_id")): item
            for item in old.get("annotations", [])
            if isinstance(item, dict) and item.get("target_id")
        }

    records = [item for item in manifest["captures"] if isinstance(item, dict)]
    if limit > 0:
        records = records[:limit]
    annotations: list[dict[str, Any]] = []
    overlays = output_path.parent / "qwen_annotated_images"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint() -> None:
        payload = {
            "schema": ANNOTATION_SCHEMA,
            "manifest": str(manifest_path.resolve()),
            "map": manifest.get("map"),
            "model_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minimum_confidence": minimum_confidence,
            "annotations": annotations,
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        write_annotation_csv(output_path.with_suffix(".csv"), annotations)

    for index, record in enumerate(records, start=1):
        target_id = str(record.get("target_id") or f"target_{index:03d}")
        images = resolve_record_images(record, directory)
        fingerprint = input_fingerprint(record, images)
        cached = previous.get(target_id)
        if cached and cached.get("input_fingerprint") == fingerprint:
            annotations.append(cached)
            print(f"[{index}/{len(records)}] 复用已有标注：{target_id}", flush=True)
            continue
        print(
            f"[{index}/{len(records)}] Qwen 标注 {target_id}，输入 {len(images)} 张照片……",
            flush=True,
        )
        raw = backend.annotate(images, build_prompt(record, len(images)))
        payload = extract_json_object(raw)
        annotation = normalise_annotation(payload, record, images, minimum_confidence)
        annotation["input_fingerprint"] = fingerprint
        annotation["raw_response"] = raw
        annotation["annotated_images"] = render_overlays(annotation, images, overlays)
        annotations.append(annotation)
        save_checkpoint()
        print(
            "  -> {} / {:.0%} / {}".format(
                annotation["semantic_label"],
                annotation["confidence"],
                "待复核" if annotation["needs_human_review"] else "已写入语义标签",
            ),
            flush=True,
        )

    save_checkpoint()
    semantic_target = semantic_output or output_path.with_name("semantic_map_qwen.json")
    merge_semantic_map(semantic_source, semantic_target, manifest, annotations)
    review_count = sum(bool(item.get("needs_human_review")) for item in annotations)
    return {
        "annotation_json": str(output_path.resolve()),
        "annotation_csv": str(output_path.with_suffix(".csv").resolve()),
        "semantic_map": str(semantic_target.resolve()),
        "annotated_images": str(overlays.resolve()),
        "count": len(annotations),
        "review_count": review_count,
    }
