from qwen_grounded_tracker.grounding.parser import parse_grounding_output


def test_parse_relative_1000_box() -> None:
    raw = '{"found":true,"bbox_2d":[100,200,600,800],"coordinate_system":"relative_1000","target_name":"cup","confidence":0.9}'
    result = parse_grounding_output(raw, 640, 480)
    assert result.found
    assert result.bbox is not None
    assert round(result.bbox.x1) == 64
    assert round(result.bbox.y1) == 96
    assert round(result.bbox.x2) == 384
    assert round(result.bbox.y2) == 384


def test_parse_markdown_json() -> None:
    raw = """```json
    {"found": true, "bbox_2d": [0.1, 0.2, 0.6, 0.8], "coordinate_system": "normalized", "target_name": "target", "confidence": 0.7}
    ```"""
    result = parse_grounding_output(raw, 1000, 500)
    assert result.found
    assert result.bbox is not None
    assert result.bbox.to_xywh() == (100, 100, 500, 300)


def test_reject_missing_box() -> None:
    result = parse_grounding_output('{"found":false,"bbox_2d":[]}', 640, 480)
    assert not result.found
    assert result.bbox is None
