"""Contracts for d1's typed answers; no model weights or GPU required."""

import importlib

import pytest
from PIL import Image


def test_d1_detect_scales_fraction_boxes_and_keeps_probabilities(monkeypatch):
    d1 = importlib.import_module("tools.d1")

    def decide(image, questions, **kwargs):
        assert questions["class_0"]["type"] == "boxes"
        return {
            "answers": {
                "class_0": {
                    "boxes": [
                        {
                            "box": {"left": 0.1, "top": 0.2, "right": 0.6, "bottom": 0.8},
                            "found": 0.9,
                            "confidence": 0.7,
                        },
                        {"box": None, "found": 0.1, "confidence": 0},
                    ]
                }
            }
        }

    monkeypatch.setattr(d1, "d1_decide", decide)
    result = d1.d1_detect(Image.new("RGB", (200, 100)), ["stop"])
    assert len(result) == 1
    assert result[0]["bbox"] == [20, 20, 120, 80]
    assert result[0]["label"] == "stop"
    assert result[0]["score"] == 0.9
    assert result[0]["box_confidence"] == 0.7


def test_judge_sends_numbered_overlay_and_preserves_rejections(monkeypatch):
    d1 = importlib.import_module("tools.d1")
    from workflows.vlm_judge import score_detections

    original = Image.new("RGB", (100, 100), "white")

    def decide(image, questions, **kwargs):
        assert image.tobytes() != original.tobytes()
        assert "#0" in questions["box_0"]["instructions"]
        return {"answers": {"box_0": {"noul": 0.8}, "box_1": {"noul": 0.2}}}

    monkeypatch.setattr(d1, "d1_decide", decide)
    detections = [{"bbox": [10, 20, 30, 40], "label": "stop"}] * 2
    result = score_detections(
        original,
        detections,
        "LiquidAI/d1-Omni-2B",
        backend="transformers",
        base_url=None,
        api_key=None,
    )
    assert [v["verdict"] for v in result] == ["correct", "incorrect"]
    assert [v["score"] for v in result] == [0.8, 0.2]


def test_vlm_detect_routes_d1_without_text_generation(monkeypatch):
    d1 = importlib.import_module("tools.d1")
    from tools.vlm_detect import vlm_detect

    monkeypatch.setattr(d1, "d1_detect", lambda *a, **kw: [{"label": "stop"}])
    assert vlm_detect(
        Image.new("RGB", (8, 8)), classes=["stop"], model_id="LiquidAI/d1-Omni-2B"
    ) == [{"label": "stop"}]
    with pytest.raises(ValueError, match="transformers"):
        vlm_detect(
            Image.new("RGB", (8, 8)),
            classes=["stop"],
            model_id="LiquidAI/d1-Omni-2B",
            backend="openai",
        )


def test_d1_judge_missing_probability_is_not_acceptance(monkeypatch):
    d1 = importlib.import_module("tools.d1")
    monkeypatch.setattr(d1, "d1_decide", lambda *a, **kw: {"answers": {}})
    result = d1.d1_judge(Image.new("RGB", (8, 8)), [{"label": "stop"}])
    assert result[0]["verdict"] == "incorrect"
    assert result[0]["score"] == 0


def test_local_d1_judge_filters_rejections_and_records_annotation_ids(tmp_path, monkeypatch):
    import json
    from workflows.vlm_judge import judge_labels

    d1 = importlib.import_module("tools.d1")
    image = Image.new("RGB", (100, 100), "white")
    image.save(tmp_path / "image.png")
    annotations = {
        "images": [{"id": 1, "file_name": "image.png", "width": 100, "height": 100}],
        "categories": [{"id": 0, "name": "stop"}],
        "annotations": [
            {"id": n, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]} for n in (10, 20)
        ],
    }
    (tmp_path / "input.json").write_text(json.dumps(annotations))
    monkeypatch.setattr(
        d1,
        "d1_decide",
        lambda *a, **kw: {"answers": {"box_0": {"noul": 0.9}, "box_1": {"noul": 0.1}}},
    )
    result = judge_labels(
        tmp_path,
        tmp_path / "judged.json",
        annotations=tmp_path / "input.json",
        model_id="LiquidAI/d1-Omni-2B",
    )
    assert [a["id"] for a in result["annotations"]] == [10]
    log = json.loads((tmp_path / "judged_verdicts.json").read_text())
    assert [v["annotation_id"] for v in log] == [10, 20]


def test_d1_review_renders_readable_labels_without_mutating_detections(monkeypatch):
    from copy import deepcopy
    import tools.bbox_viz as viz
    d1 = importlib.import_module("tools.d1")
    image = Image.new("RGB", (100, 100), "white")
    detections = [{"label": "enter_left_lane", "bbox": [10, 20, 40, 60]},
                  {"label": "red_light", "bbox": [50, 20, 80, 60]}]
    before = deepcopy(detections)
    real_draw = viz.draw_detections

    def draw(im, dets, **kwargs):
        assert [d["label"] for d in dets] == ["keep left sign", "red light"]
        assert kwargs["show_index"] is True
        return real_draw(im, dets, **kwargs)

    def decide(overlay, questions, **kwargs):
        assert overlay.tobytes() != image.tobytes()
        assert "#0, labeled keep left sign" in questions["box_0"]["instructions"]
        assert "actual keep left sign" in questions["box_0"]["instructions"]
        return {"answers": {"box_0": {"noul": .8}, "box_1": {"noul": .2}}}

    monkeypatch.setattr(viz, "draw_detections", draw)
    monkeypatch.setattr(d1, "d1_decide", decide)
    result = d1.d1_review(image, detections, label_aliases={"enter_left_lane": "keep left sign"})
    assert detections == before
    assert [v["verdict"] for v in result] == ["correct", "incorrect"]
    assert [v["detection_idx"] for v in result] == [0, 1]


def test_d1_review_empty_input_needs_no_image_or_gpu():
    from tools import d1_review
    assert d1_review("not-an-image", []) == []
