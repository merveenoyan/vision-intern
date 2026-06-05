"""RF-DETR closed-set object detection (COCO 80 classes)."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "Roboflow/rf-detr-base"


def _load() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def detect(
    image: str | Image.Image,
    threshold: float = 0.3,
) -> list[dict]:
    """Detect objects in *image* using RF-DETR.

    Returns a list of detections, each a dict with:
        label (str): COCO class name
        score (float): confidence
        box (list[float]): [x1, y1, x2, y2] in pixel coordinates
    """
    model, processor = get_or_load("rf_detr_detect", _load)
    image = load_image(image)

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=model.device)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        detections.append({
            "label": model.config.id2label[label_id.item()],
            "score": round(score.item(), 4),
            "box": [round(c, 1) for c in box.tolist()],
        })
    return detections
