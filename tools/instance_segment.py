"""RF-DETR-Seg instance segmentation (COCO 80 classes)."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "Roboflow/rf-detr-seg-large"


def _load() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForInstanceSegmentation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForInstanceSegmentation.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def instance_segment(
    image: str | Image.Image,
    threshold: float = 0.3,
) -> dict:
    """Segment instances in *image* using RF-DETR-Seg.

    Returns a dict with:
        segmentation: (H, W) int32 tensor — segment ID per pixel (-1 = background)
        segments_info: list of dicts with id (int), label (str), score (float)
    """
    model, processor = get_or_load("rf_detr_seg", _load)
    image = load_image(image)

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=threshold
    )[0]

    segments = []
    for seg in result["segments_info"]:
        segments.append({
            "id": seg["id"],
            "label": model.config.id2label[seg["label_id"]],
            "score": round(seg["score"], 4),
        })

    return {
        "segmentation": result["segmentation"],
        "segments_info": segments,
    }
