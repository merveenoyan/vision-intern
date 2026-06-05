"""Falcon-Perception — zero-shot segmentation from a text prompt."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "tiiuae/Falcon-Perception"


def _load() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    return model, None  # no separate processor


def segment_from_text(
    image: str | Image.Image,
    text: str,
) -> list[dict]:
    """Segment all instances matching *text* in the image.

    Args:
        image: input image (path, URL, or PIL Image)
        text: natural-language query, e.g. "red backpack"

    Returns a list of dicts (one per instance) with:
        center: dict with x, y (normalized 0-1)
        size: dict with h, w (normalized 0-1)
        mask_rle: COCO RLE dict at original resolution
    """
    model, _ = get_or_load("falcon_perception", _load)
    image = load_image(image)

    preds = model.generate(image, text)[0]

    results = []
    for p in preds:
        results.append({
            "center": {"x": p["xy"]["x"], "y": p["xy"]["y"]},
            "size": {"h": p["hw"]["h"], "w": p["hw"]["w"]},
            "mask_rle": {
                "counts": p["mask_rle"]["counts"],
                "size": p["mask_rle"]["size"],
            },
        })
    return results
