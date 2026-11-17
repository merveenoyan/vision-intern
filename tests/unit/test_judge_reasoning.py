"""Reasoning models must be judged using their final answer, not scratch work."""
from workflows.vlm_judge import _parse_verdicts


def test_verdict_parser_ignores_reasoning_arrays():
    text = '<think>Inspect boxes [0, 1].</think>\n[{"id": 0, "verdict": "correct", "score": 0.9}]'
    assert _parse_verdicts(text)[0]["verdict"] == "correct"


def test_verdict_parser_rejects_unfinished_reasoning():
    text = '<think>Possible answer: [{"id": 0, "verdict": "correct", "score": 0.9}]'
    assert _parse_verdicts(text) == {}


def test_lfm3_judge_has_room_for_reasoning(monkeypatch):
    from PIL import Image
    from workflows import vlm_judge

    def worker(*args, max_tokens, **kwargs):
        # Simulate a model needing more than the old 1024-token budget to answer.
        if max_tokens < 16384:
            return '<think>Still inspecting the image'
        return '</think>[{"id": 0, "verdict": "correct", "score": 0.9}]'

    monkeypatch.setattr(vlm_judge, "run_vlm", worker)
    image = Image.new("RGB", (8, 8))
    result = vlm_judge.score_detections(
        image, [{"label": "cat", "bbox": [0, 0, 8, 8]}],
        model_id="LiquidAI/LFM3-VL-2B", overlay_img=image,
        backend="transformers", base_url=None, api_key=None,
    )
    assert result[0]["verdict"] == "correct"
