"""MM-Grounding-DINO — open-vocabulary object detection from text queries."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"


def _load() -> tuple[Any, Any]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def grounded_detect(
    image: str | Image.Image,
    text_queries: list[str],
    threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[dict]:
    """Detect objects matching free-form *text_queries*.

    Args:
        image: input image (path, URL, or PIL Image)
        text_queries: open-vocabulary class names, e.g. ["cat", "remote control"]
        threshold: detection confidence threshold
        text_threshold: per-token threshold for phrase extraction

    Returns a list of detections, each a dict with:
        label (str): matched text phrase
        score (float): confidence
        box (list[float]): [x1, y1, x2, y2] in pixel coordinates
    """
    model, processor = get_or_load("mm_grounding_dino", _load)
    image = load_image(image)

    inputs = processor(
        images=image, text=[text_queries], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        detections.append({
            "label": label,
            "score": round(score.item(), 4),
            "box": [round(c, 1) for c in box.tolist()],
        })
    return detections
