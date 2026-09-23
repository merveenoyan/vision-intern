import pytest

from tools.dataset_utils import resolve_annotation_source


def test_explicit_detections_source_wins_when_objects_also_exists():
    columns = ["image", "objects", "detections"]

    assert resolve_annotation_source(columns, "detections", "detections") == "detections"


def test_requested_annotation_source_must_exist():
    with pytest.raises(ValueError, match="detections"):
        resolve_annotation_source(["image", "objects"], "detections", "detections")
