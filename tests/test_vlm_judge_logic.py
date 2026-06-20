"""Tests for pure-Python judge logic in workflows/vlm_judge.py.

Covers _parse_verdicts, ensemble_row, and score_detections — all exercisable
without a GPU or a live VLM endpoint.  GPU-heavy imports are stubbed in
tests/conftest.py so the module can be imported on any machine.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from workflows.vlm_judge import _parse_verdicts, ensemble_row, score_detections


# ---------------------------------------------------------------------------
# _parse_verdicts
# ---------------------------------------------------------------------------

class TestParseVerdicts:
    def test_basic_int_id(self):
        text = '[{"id": 0, "verdict": "correct", "score": 0.9, "reason": "tight box"}]'
        result = _parse_verdicts(text)
        assert result == {0: {"verdict": "correct", "score": 0.9, "reason": "tight box"}}

    def test_multiple_detections(self):
        text = json.dumps([
            {"id": 0, "verdict": "correct", "score": 0.9, "reason": "good"},
            {"id": 1, "verdict": "incorrect", "score": 0.1, "reason": "empty region"},
        ])
        result = _parse_verdicts(text)
        assert result[0]["verdict"] == "correct"
        assert result[1]["verdict"] == "incorrect"
        assert len(result) == 2

    def test_string_id_hash_prefix(self):
        text = '[{"id": "#0", "verdict": "correct", "score": 0.8, "reason": "ok"}]'
        result = _parse_verdicts(text)
        assert 0 in result
        assert result[0]["verdict"] == "correct"

    def test_string_id_box_label(self):
        text = '[{"id": "box 1", "verdict": "incorrect", "score": 0.2, "reason": "bad"}]'
        result = _parse_verdicts(text)
        assert 1 in result
        assert result[1]["verdict"] == "incorrect"

    def test_string_id_no_digits_skipped(self):
        text = '[{"id": "no-number-here", "verdict": "correct", "score": 0.9, "reason": ""}]'
        result = _parse_verdicts(text)
        assert result == {}

    def test_float_id_converted(self):
        text = '[{"id": 1.0, "verdict": "correct", "score": 0.75, "reason": "ok"}]'
        result = _parse_verdicts(text)
        assert 1 in result

    def test_missing_id_skipped(self):
        text = '[{"verdict": "correct", "score": 0.9, "reason": "no id"}]'
        result = _parse_verdicts(text)
        assert result == {}

    def test_missing_score_defaults_zero(self):
        text = '[{"id": 0, "verdict": "correct", "reason": "present"}]'
        result = _parse_verdicts(text)
        assert result[0]["score"] == 0.0

    def test_missing_verdict_defaults_incorrect(self):
        text = '[{"id": 0, "score": 0.9, "reason": "something"}]'
        result = _parse_verdicts(text)
        assert result[0]["verdict"] == "incorrect"

    def test_missing_reason_defaults_empty(self):
        text = '[{"id": 0, "verdict": "correct", "score": 0.9}]'
        result = _parse_verdicts(text)
        assert result[0]["reason"] == ""

    def test_non_dict_item_skipped(self):
        text = '[42, {"id": 0, "verdict": "correct", "score": 0.9, "reason": "ok"}]'
        result = _parse_verdicts(text)
        assert len(result) == 1
        assert 0 in result

    def test_malformed_json_returns_empty(self):
        result = _parse_verdicts("this is not JSON at all")
        assert result == {}

    def test_no_array_returns_empty(self):
        result = _parse_verdicts('The objects look {"verdict": "correct"}')
        assert result == {}

    def test_empty_array_returns_empty(self):
        result = _parse_verdicts("[]")
        assert result == {}

    def test_fenced_code_block(self):
        text = '```json\n[{"id": 0, "verdict": "correct", "score": 0.85, "reason": "ok"}]\n```'
        result = _parse_verdicts(text)
        assert 0 in result
        assert result[0]["verdict"] == "correct"

    def test_prose_around_array(self):
        text = 'Based on my analysis:\n[{"id": 0, "verdict": "correct", "score": 0.7, "reason": "fine"}]\nDone.'
        result = _parse_verdicts(text)
        assert 0 in result

    def test_imprecise_verdict_preserved(self):
        text = '[{"id": 0, "verdict": "imprecise", "score": 0.5, "reason": "slightly off"}]'
        result = _parse_verdicts(text)
        assert result[0]["verdict"] == "imprecise"

    def test_duplicate_id_last_wins(self):
        text = json.dumps([
            {"id": 0, "verdict": "incorrect", "score": 0.1, "reason": "first"},
            {"id": 0, "verdict": "correct", "score": 0.9, "reason": "second"},
        ])
        result = _parse_verdicts(text)
        assert result[0]["verdict"] == "correct"


# ---------------------------------------------------------------------------
# ensemble_row
# ---------------------------------------------------------------------------

def _make_verdict(idx: int, verdict: str, score: float, reason: str = "") -> dict:
    return {"detection_idx": idx, "verdict": verdict, "score": score, "reason": reason}


class TestEnsembleRow:
    def test_empty_n_dets(self):
        result = ensemble_row({"gemma": [], "lfm": []}, n_dets=0, min_agree=1)
        assert result == []

    def test_both_judges_correct_min_agree_2(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.8)],
            "lfm":   [_make_verdict(0, "correct", 0.9)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=2)
        assert len(result) == 1
        r = result[0]
        assert r["detection_idx"] == 0
        assert r["n_correct"] == 2
        assert r["ensemble_keep"] is True
        assert r["mean_score"] == pytest.approx(0.85, abs=1e-4)

    def test_one_correct_one_incorrect_min_agree_2_keeps_false(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct",   0.8)],
            "lfm":   [_make_verdict(0, "incorrect", 0.2)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=2)
        assert result[0]["ensemble_keep"] is False
        assert result[0]["n_correct"] == 1

    def test_one_correct_one_incorrect_min_agree_1_keeps_true(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct",   0.8)],
            "lfm":   [_make_verdict(0, "incorrect", 0.2)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=1)
        assert result[0]["ensemble_keep"] is True

    def test_imprecise_does_not_count_as_correct(self):
        per_judge = {
            "gemma": [_make_verdict(0, "imprecise", 0.6)],
            "lfm":   [_make_verdict(0, "correct",   0.9)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=2)
        assert result[0]["n_correct"] == 1
        assert result[0]["ensemble_keep"] is False

    def test_missing_detection_idx_uses_no_verdict(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.8), _make_verdict(1, "correct", 0.7)],
            "lfm":   [_make_verdict(0, "correct", 0.9)],  # det 1 missing from lfm
        }
        result = ensemble_row(per_judge, n_dets=2, min_agree=1)
        # det 0: both correct → keep
        assert result[0]["ensemble_keep"] is True
        assert result[0]["n_correct"] == 2
        # det 1: gemma correct, lfm falls back to _NO_VERDICT (incorrect) → n_correct=1
        assert result[1]["n_correct"] == 1
        assert result[1]["ensemble_keep"] is True  # 1 >= min_agree=1

    def test_empty_per_judge_row(self):
        result = ensemble_row({}, n_dets=1, min_agree=1)
        assert result[0]["n_correct"] == 0
        assert result[0]["mean_score"] == 0.0
        assert result[0]["ensemble_keep"] is False  # 0 < 1

    def test_mean_score_calculation(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.6)],
            "lfm":   [_make_verdict(0, "correct", 0.4)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=1)
        assert result[0]["mean_score"] == pytest.approx(0.5, abs=1e-4)

    def test_three_judges_two_agree(self):
        per_judge = {
            "a": [_make_verdict(0, "correct",   0.9)],
            "b": [_make_verdict(0, "correct",   0.8)],
            "c": [_make_verdict(0, "incorrect", 0.1)],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=2)
        assert result[0]["n_correct"] == 2
        assert result[0]["ensemble_keep"] is True

    def test_per_judge_dict_structure(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.8, "looks good")],
        }
        result = ensemble_row(per_judge, n_dets=1, min_agree=1)
        pj = result[0]["per_judge"]
        assert "gemma" in pj
        assert pj["gemma"]["verdict"] == "correct"
        assert pj["gemma"]["score"] == pytest.approx(0.8)
        assert pj["gemma"]["reason"] == "looks good"

    def test_multiple_detections_independent(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.9), _make_verdict(1, "incorrect", 0.1)],
            "lfm":   [_make_verdict(0, "correct", 0.8), _make_verdict(1, "incorrect", 0.2)],
        }
        result = ensemble_row(per_judge, n_dets=2, min_agree=2)
        assert result[0]["ensemble_keep"] is True
        assert result[1]["ensemble_keep"] is False

    def test_detection_idx_preserved(self):
        per_judge = {
            "gemma": [_make_verdict(0, "correct", 0.9), _make_verdict(1, "correct", 0.8)],
        }
        result = ensemble_row(per_judge, n_dets=2, min_agree=1)
        assert result[0]["detection_idx"] == 0
        assert result[1]["detection_idx"] == 1


# ---------------------------------------------------------------------------
# score_detections
# ---------------------------------------------------------------------------

class TestScoreDetections:
    def test_empty_detections_returns_empty(self):
        result = score_detections(
            MagicMock(), [], "some/model",
            backend="openai", base_url=None, api_key=None,
        )
        assert result == []

    def test_returns_one_entry_per_detection(self):
        dets = [
            {"label": "table", "bbox": [10, 10, 100, 100]},
            {"label": "image", "bbox": [200, 200, 300, 300]},
        ]
        resp = json.dumps([
            {"id": 0, "verdict": "correct",   "score": 0.9, "reason": "ok"},
            {"id": 1, "verdict": "incorrect", "score": 0.1, "reason": "bad"},
        ])
        with patch("workflows.vlm_judge.run_vlm", return_value=resp):
            result = score_detections(
                MagicMock(), dets, "some/model",
                backend="openai", base_url=None, api_key=None,
            )
        assert len(result) == 2
        assert result[0]["detection_idx"] == 0
        assert result[0]["verdict"] == "correct"
        assert result[1]["detection_idx"] == 1
        assert result[1]["verdict"] == "incorrect"

    def test_vlm_exception_falls_back_to_no_verdict(self):
        dets = [{"label": "table", "bbox": [0, 0, 50, 50]}]
        with patch("workflows.vlm_judge.run_vlm", side_effect=RuntimeError("timeout")):
            result = score_detections(
                MagicMock(), dets, "some/model",
                backend="openai", base_url=None, api_key=None,
            )
        assert len(result) == 1
        assert result[0]["verdict"] == "incorrect"
        assert result[0]["score"] == 0.0

    def test_malformed_response_falls_back_to_no_verdict(self):
        dets = [{"label": "chart", "bbox": [5, 5, 50, 50]}]
        with patch("workflows.vlm_judge.run_vlm", return_value="not json"):
            result = score_detections(
                MagicMock(), dets, "some/model",
                backend="openai", base_url=None, api_key=None,
            )
        assert result[0]["verdict"] == "incorrect"
        assert result[0]["score"] == 0.0

    def test_prebuilt_overlay_used_as_is(self):
        # When overlay_img is supplied, score_detections must use it directly
        # and must NOT call draw_detections (the local import inside the branch).
        dets = [{"label": "table", "bbox": [10, 10, 100, 100]}]
        from PIL import Image as PILImage
        overlay = PILImage.new("RGB", (100, 100), (0, 0, 0))
        resp = json.dumps([{"id": 0, "verdict": "correct", "score": 0.8, "reason": "ok"}])
        with patch("workflows.vlm_judge.run_vlm", return_value=resp) as mock_vlm:
            result = score_detections(
                MagicMock(), dets, "m",
                backend="openai", base_url=None, api_key=None,
                overlay_img=overlay,
            )
        # vlm was called with the overlay we provided (not a newly rendered one)
        call_img_arg = mock_vlm.call_args[0][0]
        assert call_img_arg is overlay
        assert result[0]["verdict"] == "correct"

    def test_no_overlay_triggers_draw(self):
        # draw_detections is imported locally inside score_detections; patch at source.
        dets = [{"label": "table", "bbox": [10, 10, 100, 100]}]
        resp = json.dumps([{"id": 0, "verdict": "correct", "score": 0.8, "reason": "ok"}])
        from PIL import Image as PILImage
        fake_overlay = PILImage.new("RGB", (100, 100), (255, 0, 0))
        with patch("workflows.vlm_judge.run_vlm", return_value=resp), \
             patch("tools.bbox_viz.draw_detections", return_value=fake_overlay) as mock_draw:
            score_detections(
                MagicMock(), dets, "m",
                backend="openai", base_url=None, api_key=None,
            )
        mock_draw.assert_called_once()
