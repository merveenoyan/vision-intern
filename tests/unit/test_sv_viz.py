"""Unit tests for the supervision/trackers visualization layer.

The detection-dict <-> ``supervision.Detections`` bridge, the bbox annotator,
and the tracker name resolution. Tests that need ``supervision`` / ``trackers``
are skipped when the ``viz`` extra is not installed; the registry/wiring tests
run regardless and never import torch.
"""

import subprocess
import sys

import pytest

sv = pytest.importorskip("supervision")

from tools.sv_convert import from_supervision, to_supervision  # noqa: E402
from tools.sv_viz import _labels, annotate  # noqa: E402

DETS = [
    {"label": "car", "score": 0.9, "box": [10, 20, 100, 120]},
    {"label": "person", "score": 0.8, "bbox": [50, 60, 90, 200], "sub_label": "pedestrian"},
    {"label": "car", "box": [200, 10, 260, 80], "track_id": 7},
    {"label": "malformed", "box": [1, 2, 3]},  # dropped: not 4 coords
]
VERDICTS = [{"detection_idx": 0, "mean_score": 0.75}, {"detection_idx": 1, "score": 0.4}]


# ------------------------------------------------------------------
# Converter
# ------------------------------------------------------------------

def test_to_supervision_drops_malformed_and_maps_fields():
    d = to_supervision(DETS, verdicts=VERDICTS)
    assert len(d) == 3                                  # malformed box dropped
    assert list(d.class_id) == [0, 1, 0]               # stable label -> id
    assert list(d.data["class_name"]) == ["car", "person", "car"]
    assert list(d.confidence)[:2] == [0.9, 0.8]
    assert list(d.tracker_id) == [-1, -1, 7]           # missing -> sentinel -1


def test_verdict_scores_align_to_original_indices():
    d = to_supervision(DETS, verdicts=VERDICTS)
    import numpy as np
    assert d.data["verdict_score"][0] == pytest.approx(0.75)
    assert d.data["verdict_score"][1] == pytest.approx(0.40)
    assert np.isnan(d.data["verdict_score"][2])         # box 2 had no verdict


def test_class_map_is_shared_for_stable_colors():
    cmap = {}
    to_supervision([{"label": "a", "box": [0, 0, 1, 1]}], class_map=cmap)
    d2 = to_supervision([{"label": "b", "box": [0, 0, 1, 1]},
                         {"label": "a", "box": [0, 0, 1, 1]}], class_map=cmap)
    assert cmap == {"a": 0, "b": 1}
    assert list(d2.class_id) == [1, 0]                  # 'a' keeps id 0 across calls


def test_empty_detections():
    assert to_supervision([]).is_empty()
    assert from_supervision(to_supervision([])) == []


def test_roundtrip_preserves_track_and_sublabel_and_drops_sentinel():
    rt = from_supervision(to_supervision(DETS))
    assert rt[1]["sub_label"] == "pedestrian"
    assert rt[2]["track_id"] == 7
    assert "track_id" not in rt[0]                      # -1 sentinel omitted


# ------------------------------------------------------------------
# Labels + annotate
# ------------------------------------------------------------------

def test_labels_prefix_with_tracker_id_then_index():
    d = to_supervision(DETS, verdicts=VERDICTS)
    labels = _labels(d, show_index=True, show_conf=False)
    # tracker_id present -> '#<id>' prefix wins over index; verdict appended
    assert labels == ["#-1 car 0.75", "#-1 person 0.40", "#7 car"]


def test_labels_index_prefix_without_tracking():
    d = to_supervision([{"label": "car", "score": 0.9, "box": [0, 0, 1, 1]}])
    assert _labels(d, show_index=True, show_conf=True) == ["#0 car 0.90"]
    assert _labels(d, show_index=False, show_conf=False) == ["car"]


def test_annotate_returns_same_size_image():
    from PIL import Image
    img = Image.new("RGB", (320, 240), "gray")
    out = annotate(img, DETS, verdicts=VERDICTS, show_index=True)
    assert out.size == (320, 240) and out.mode == "RGB"
    assert out.tobytes() != img.tobytes()               # something was drawn


# ------------------------------------------------------------------
# Instance masks (segmentation tracking path)
# ------------------------------------------------------------------

def _mask(h, w, y0, y1, x0, x1):
    import numpy as np
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_masks_stacked_when_all_present():
    d = to_supervision([
        {"label": "car", "box": [5, 5, 20, 15], "mask": _mask(40, 60, 5, 15, 5, 20)},
        {"label": "person", "box": [30, 20, 50, 30], "mask": _mask(40, 60, 20, 30, 30, 50)},
    ])
    assert d.mask is not None and d.mask.shape == (2, 40, 60) and d.mask.dtype == bool


def test_masks_dropped_when_partial_or_mismatched():
    # one box has no mask -> can't stack -> no mask array
    assert to_supervision([
        {"label": "a", "box": [0, 0, 2, 2], "mask": _mask(10, 10, 0, 2, 0, 2)},
        {"label": "b", "box": [1, 1, 3, 3]},
    ]).mask is None
    # differing mask shapes -> no mask array
    assert to_supervision([
        {"label": "a", "box": [0, 0, 2, 2], "mask": _mask(10, 10, 0, 2, 0, 2)},
        {"label": "b", "box": [1, 1, 3, 3], "mask": _mask(12, 12, 1, 3, 1, 3)},
    ]).mask is None


def test_mask_roundtrips():
    src = [{"label": "car", "box": [5, 5, 20, 15], "mask": _mask(40, 60, 5, 15, 5, 20)}]
    rt = from_supervision(to_supervision(src))
    assert rt[0]["mask"].shape == (40, 60)


def test_annotate_draws_masks():
    from PIL import Image
    dets = [{"label": "car", "box": [5, 5, 20, 15], "mask": _mask(48, 64, 5, 15, 5, 20)}]
    out = annotate(Image.new("RGB", (64, 48), "gray"), dets)
    assert out.size == (64, 48)
    assert out.tobytes() != Image.new("RGB", (64, 48), "gray").tobytes()


def test_seg_adapter_derives_boxes_and_masks():
    import numpy as np
    from tools.track_video import _seg_to_dets
    seg = np.full((48, 64), -1, np.int32)
    seg[5:15, 5:20] = 1
    seg[20:40, 30:50] = 2
    result = {"segmentation": seg, "segments_info": [
        {"id": 1, "label": "car", "score": 0.9},
        {"id": 2, "label": "person", "score": 0.8},
    ]}
    out = _seg_to_dets(result)
    assert out[0]["box"] == [5.0, 5.0, 20.0, 15.0]
    assert out[0]["label"] == "car" and out[0]["mask"].sum() == 10 * 15


def test_falcon_adapter_box_from_mask_and_fallback(monkeypatch):
    import numpy as np
    import tools.utils as u
    from tools.track_video import _falcon_to_dets

    real = np.zeros((40, 50), bool)
    real[10:20, 5:25] = True
    seq = [real, np.zeros((40, 50), bool)]            # 2nd empty -> center/size fallback
    monkeypatch.setattr(u, "rle_to_mask", lambda rle: seq[rle["_i"]])
    preds = [
        {"center": {"x": 0.3, "y": 0.375}, "size": {"w": 0.4, "h": 0.25}, "mask_rle": {"_i": 0}},
        {"center": {"x": 0.5, "y": 0.5}, "size": {"w": 0.2, "h": 0.2}, "mask_rle": {"_i": 1}},
    ]
    out = _falcon_to_dets(preds, "red car", 50, 40)
    assert out[0]["box"] == [5.0, 10.0, 25.0, 20.0] and out[0]["label"] == "red car"
    assert [round(c, 1) for c in out[1]["box"]] == [20.0, 16.0, 30.0, 24.0]


# ------------------------------------------------------------------
# Tracker resolution (needs the `trackers` package)
# ------------------------------------------------------------------

def test_tracker_resolution():
    pytest.importorskip("trackers")
    from tools.track_video import _resolve_tracker
    assert type(_resolve_tracker("bytetrack")).__name__ == "ByteTrackTracker"
    assert type(_resolve_tracker("sort")).__name__ == "SORTTracker"
    assert type(_resolve_tracker("ocsort")).__name__ == "OCSORTTracker"
    with pytest.raises(ValueError):
        _resolve_tracker("not-a-tracker")


# ------------------------------------------------------------------
# Registry wiring (runs without the viz extra; must not import torch)
# ------------------------------------------------------------------

def test_track_video_registered_as_train_tool():
    from tools.registry import list_tools
    assert "track_video" in list_tools(include_train=True)
    assert "track_video" not in list_tools(include_train=False)


def test_light_path_does_not_import_supervision_or_torch():
    code = (
        "import sys, tools; "
        "tools.as_json_schema(); tools.list_tools(); "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "assert 'supervision' not in sys.modules, 'supervision leaked'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=".")
