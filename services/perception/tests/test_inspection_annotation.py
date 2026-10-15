from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from qwen_grounded_tracker.inspection_annotation import (
    discover_manifest,
    run_annotation,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def annotate(self, image_paths, prompt):
        self.calls += 1
        assert len(image_paths) == 2
        assert "target_id: object_1" in prompt
        return json.dumps(
            {
                "semantic_label": "table",
                "name_zh": "桌子",
                "category": "furniture",
                "description_zh": "黑色矩形桌体",
                "visible_attributes": ["黑色", "矩形"],
                "condition": "normal",
                "fixed_or_temporary": "probably_fixed",
                "same_entity_across_views": True,
                "confidence": 0.88,
                "evidence_quality": "good",
                "needs_human_review": False,
                "review_reason": "",
                "views": [
                    {
                        "image_index": 1,
                        "target_visible": True,
                        "bbox_2d": [100, 100, 800, 900],
                        "confidence": 0.9,
                    },
                    {
                        "image_index": 2,
                        "target_visible": True,
                        "bbox_2d": [150, 120, 850, 920],
                        "confidence": 0.86,
                    },
                ],
            },
            ensure_ascii=False,
        )


def _make_inspection(tmp_path: Path) -> Path:
    root = tmp_path / "inspections" / "20260813_120000"
    root.mkdir(parents=True)
    Image.new("RGB", (160, 120), "white").save(root / "object.jpg")
    Image.new("RGB", (160, 120), "gray").save(root / "object_camera_2.jpg")
    manifest = {
        "map": "maps/ble_scene_17.pgm",
        "captures": [
            {
                "target_id": "object_1",
                "kind": "interior_black_region",
                "image": "object.jpg",
                "camera_2_image": "object_camera_2.jpg",
                "target_x_m": 1.2,
                "target_y_m": -0.4,
                "inspection_subject": "unknown_mapped_black_region",
                "review_question": "确认物体类型",
            }
        ],
    }
    path = root / "inspection_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def test_annotation_dual_camera_checkpoint_and_semantic_merge(tmp_path: Path):
    manifest = _make_inspection(tmp_path)
    source_semantic = tmp_path / "semantics.json"
    source_semantic.write_text(
        json.dumps({"format_version": 1, "entities": {}}), encoding="utf-8"
    )
    output = manifest.parent / "qwen_annotations.json"
    semantic_output = manifest.parent / "semantic_map_qwen.json"
    backend = FakeBackend()

    result = run_annotation(
        manifest,
        backend,
        output,
        source_semantic,
        semantic_output,
    )
    assert result["count"] == 1
    assert result["review_count"] == 0
    assert backend.calls == 1
    annotation = json.loads(output.read_text(encoding="utf-8"))["annotations"][0]
    assert annotation["semantic_label"] == "table"
    assert len(annotation["annotated_images"]) == 2
    merged = json.loads(semantic_output.read_text(encoding="utf-8"))
    assert merged["entities"]["object_1"]["semantic_label"] == "table"
    assert json.loads(source_semantic.read_text(encoding="utf-8"))["entities"] == {}

    run_annotation(manifest, backend, output, source_semantic, semantic_output)
    assert backend.calls == 1


def test_discover_manifest_uses_direct_or_latest(tmp_path: Path):
    manifest = _make_inspection(tmp_path)
    assert discover_manifest(manifest.parent) == manifest.resolve()
    assert discover_manifest(tmp_path / "inspections") == manifest.resolve()
