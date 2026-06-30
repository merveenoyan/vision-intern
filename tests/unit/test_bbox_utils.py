"""Smoke tests for the CPU-only bbox utilities (no model, no network)."""

import math

import pytest

from tools.bbox_utils import (
    BBOX_FORMATS,
    compute_stats,
    convert_annotations,
    convert_bbox,
    validate_annotations,
)


@pytest.mark.parametrize("fmt", BBOX_FORMATS)
def test_convert_roundtrips_through_every_format(fmt):
    """coco_xywh -> fmt -> coco_xywh must recover the original box."""
    img_w, img_h = 640, 480
    original = [10.0, 20.0, 100.0, 50.0]  # coco_xywh
    forward = convert_bbox(original, "coco_xywh", fmt, img_w, img_h)
    back = convert_bbox(forward, fmt, "coco_xywh", img_w, img_h)
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(original, back)), (
        f"round-trip via {fmt} drifted: {original} -> {forward} -> {back}"
    )


def test_convert_known_values():
    # coco_xywh -> xyxy is just [x, y, x+w, y+h]
    assert convert_bbox([10, 20, 100, 50], "coco_xywh", "xyxy") == [10, 20, 110, 70]
    # xyxy -> yolo (normalised centre + size) on a 100x100 image
    assert convert_bbox([0, 0, 50, 50], "xyxy", "yolo", 100, 100) == [0.25, 0.25, 0.5, 0.5]


def test_convert_rejects_unknown_format():
    with pytest.raises(ValueError):
        convert_bbox([0, 0, 1, 1], "bogus", "xyxy")


def test_convert_annotations_preserves_other_fields():
    anns = [{"bbox": [10, 20, 100, 50], "category_id": 3, "score": 0.9}]
    out = convert_annotations(anns, "coco_xywh", "xyxy")
    assert out[0]["bbox"] == [10, 20, 110, 70]
    assert out[0]["category_id"] == 3 and out[0]["score"] == 0.9


def test_validate_flags_bad_boxes():
    anns = [
        {"bbox": [0, 0, 10, 10], "category_id": 1},   # ok
        {"bbox": [10, 10], "category_id": 1},          # E002 too few values
        {"bbox": [0, 0, float("inf"), 10], "category_id": 1},  # E003 non-finite
        {"bbox": [50, 0, 10, 10], "category_id": 1, "_fmt": "xyxy"},  # see below
    ]
    issues = validate_annotations(anns[:3], bbox_format="coco_xywh")
    codes = {i["code"] for i in issues}
    assert "E002" in codes and "E003" in codes


def test_validate_detects_inverted_xyxy():
    # xmin > xmax in xyxy space -> E004
    issues = validate_annotations([{"bbox": [50, 0, 10, 10], "category_id": 1}],
                                  bbox_format="xyxy")
    assert any(i["code"] == "E004" for i in issues)


def test_validate_clean_annotations_have_no_errors():
    anns = [{"bbox": [0, 0, 10, 10], "category_id": 1}]
    issues = validate_annotations(anns, bbox_format="coco_xywh", img_w=100, img_h=100)
    assert not [i for i in issues if i["level"] == "error"]


def test_compute_stats_on_minimal_coco():
    coco = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 1, "name": "table"}, {"id": 2, "name": "figure"}],
    }
    stats = compute_stats(coco)
    assert stats["total_images"] == 1
    assert stats["total_annotations"] == 2
    assert stats["unique_categories"] == 2
    assert set(stats["label_distribution"]) == {"table", "figure"}
    # the two categories co-occur in the one image
    assert stats["co_occurrence_pairs"][0]["count"] == 1
