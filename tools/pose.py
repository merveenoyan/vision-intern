"""Sapiens2 — dense human pose estimation (308 keypoints)."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "facebook/sapiens2-pose-0.4b"


def _load() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForPoseEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForPoseEstimation.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def estimate_pose(
    image: str | Image.Image,
    bboxes: list[list[float]],
    threshold: float = 0.3,
) -> list[dict]:
    """Estimate dense pose keypoints for each person bounding box.

    Args:
        image: input image (path, URL, or PIL Image)
        bboxes: list of [x, y, w, h] person boxes in COCO format (absolute pixels)
        threshold: minimum keypoint confidence to keep (lower-confidence points
                   are still returned but with score below this value)

    Returns a list (one per person) of dicts with:
        keypoints: (K, 2) float array of (x, y) in original image pixels
        scores: (K,) float array of per-keypoint confidence
        bbox: [x1, y1, x2, y2] in pixels
    """
    model, processor = get_or_load("sapiens2_pose", _load)
    image = load_image(image)

    # Sapiens2ImageProcessor expects boxes as [image][person][x,y,w,h]
    inputs = processor(image, boxes=[bboxes], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_pose_estimation(
        outputs, boxes=[bboxes], threshold=threshold
    )

    persons = []
    for person in results[0]:
        persons.append({
            "keypoints": person["keypoints"].cpu().numpy(),
            "scores": person["scores"].cpu().numpy(),
            "bbox": person["bbox"].cpu().tolist(),
        })
    return persons
