"""Liquid System One adapters: typed decisions, boxes, and overlay judgments.

Uses AutoModel.system_one rather than text generation. Imports stay torch-free
until inference, and outputs use the harness's plain detection/verdict dicts.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image

from .utils import get_or_load, load_image

DEFAULT_MODEL = "LiquidAI/d1-Omni-2B"


def is_d1(model_id: str) -> bool:
    return model_id.lower().split("/")[-1].startswith("d1-")


def _load(model_id: str):
    import torch
    from transformers import AutoModel

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
    return model.to(device).eval(), None


def d1_decide(
    image: str | Image.Image | None,
    questions: dict,
    state: Any = None,
    model_id: str = DEFAULT_MODEL,
) -> dict:
    """Answer named typed questions over an image and/or text/JSON state.

    Parameters
    ----------
    image : str or PIL.Image, optional
        Image path/URL or PIL image; None for text-only extraction.
    questions : dict
        Map each output name to a question with "type" and "instructions".
        Use text, number, or date to extract values from state; noul for
        yes/no (returns P(yes)); choice with a "criteria" mapping of option
        names to descriptions; boxes with "max" to locate image objects.
        Example: {"total": {"type": "number", "instructions": "Total due?"}}.
    state : object, optional
        Text or JSON context, including document text for field extraction.
        Text/number/date extract spans from this state, not image pixels.
    model_id : str
        System One model checkpoint.

    Returns
    -------
    dict
        Native answers and usage; output_tokens is zero. Extraction answers
        contain their typed value, span and found probability. A missing value
        is None. The found probability measures presence, not correctness.
    """
    import torch

    model, _ = get_or_load(f"d1:{model_id}", lambda: _load(model_id))
    media = [load_image(image)] if image is not None else None
    with torch.inference_mode():
        return model.system_one(state, questions, media=media)


def d1_detect(
    image: str | Image.Image,
    classes: list[str],
    model_id: str = DEFAULT_MODEL,
    class_descriptions: dict[str, str] | None = None,
    max_instances: int = 16,
) -> list[dict]:
    """Locate each requested class; convert normalized edges to pixel xyxy boxes."""
    if not classes:
        raise ValueError("d1 detection requires explicit classes")
    if not 1 <= max_instances <= 64:
        raise ValueError("max_instances must be between 1 and 64")
    image = load_image(image)
    descriptions = class_descriptions or {}
    questions = {
        f"class_{i}": {
            "type": "boxes",
            "instructions": f"Every {descriptions.get(label, label.replace('_', ' '))}.",
            "max": max_instances,
        }
        for i, label in enumerate(classes)
    }
    result = d1_decide(image, questions, model_id=model_id)
    detections = []
    w, h = image.size
    for i, label in enumerate(classes):
        answer = result["answers"][f"class_{i}"]
        for item in answer.get("boxes", []):
            box = item.get("box")
            if box is None:
                continue
            coords = [float(box[k]) for k in ("left", "top", "right", "bottom")]
            if not all(math.isfinite(v) for v in coords):
                continue
            bbox = [round(max(0, min(1, v)) * scale) for v, scale in zip(coords, (w, h, w, h))]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "sub_label": "",
                    "score": float(item["found"]),
                    "box_confidence": float(item["confidence"]),
                    "at_least": bool(answer.get("at_least", False)),
                }
            )
    return detections


def d1_judge(
    overlay: str | Image.Image,
    detections: list[dict],
    model_id: str = DEFAULT_MODEL,
    class_descriptions: dict[str, str] | None = None,
) -> list[dict]:
    """Judge numbered box overlays; score is P(correct class AND tight box)."""
    if not detections:
        return []
    definitions = class_descriptions or {}
    questions = {}
    for i, det in enumerate(detections):
        label = det["label"]
        definition = definitions.get(label, label.replace("_", " "))
        questions[f"box_{i}"] = {
            "type": "noul",
            "instructions": (
                f"Inspect the drawn bounding box numbered #{i}, labeled {label}. "
                f"Is this box tightly enclosing one actual {definition}, "
                "with the correct class and without cutting off the object or "
                "including substantial unrelated background? Judge the image content, "
                "not whether the printed label states the class."
            ),
        }
    result = d1_decide(overlay, questions, model_id=model_id)
    verdicts = []
    for i in range(len(detections)):
        value = result.get("answers", {}).get(f"box_{i}", {}).get("noul")
        valid = isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1
        score = float(value) if valid else 0.0
        verdicts.append(
            {
                "detection_idx": i,
                "score": score,
                "verdict": "correct" if valid and score >= 0.5 else "incorrect",
                "reason": "System One P(correct class and tight box)"
                if valid
                else "missing or invalid System One probability",
            }
        )
    return verdicts


def d1_review(
    image: str | Image.Image,
    detections: list[dict],
    label_aliases: dict[str, str] | None = None,
    model_id: str = DEFAULT_MODEL,
) -> list[dict]:
    """Review boxes with d1 using readable overlays and no class descriptions.

    Parameters
    ----------
    image : str or PIL.Image
        Original image path/URL or PIL image, without existing box overlays.
    detections : list[dict]
        Proposed detections with label and pixel-xyxy bbox fields, in box order.
    label_aliases : dict[str, str], optional
        Original label to readable display name. Unmapped labels have underscores
        replaced by spaces. Supply aliases for abbreviations or dataset codes.
    model_id : str
        System One judge checkpoint; defaults to LiquidAI/d1-Omni-2B.

    Returns
    -------
    list[dict]
        One detection_idx, score, verdict and reason per input box. Rejections
        are review flags: no detections are removed or modified. Score is
        P(correct class and tight box), accepted at >= 0.5. Missing/invalid
        probabilities produce an incorrect verdict with an explanatory reason.
    """
    if not detections:
        return []
    from .bbox_viz import draw_detections

    aliases = label_aliases or {}
    displayed = [
        {**det, "label": aliases.get(det["label"], det["label"].replace("_", " "))}
        for det in detections
    ]
    overlay = draw_detections(load_image(image), displayed, show_index=True)
    return d1_judge(overlay, displayed, model_id=model_id, class_descriptions={})
