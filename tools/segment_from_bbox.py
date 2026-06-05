"""SAM3 Tracker — refine bounding boxes into high-quality masks.

Note: facebook/sam3 is a gated model. Run `huggingface-cli login` and request
access at https://huggingface.co/facebook/sam3 before first use.
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "facebook/sam3"


def _load() -> tuple[Any, Any]:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    model = Sam3TrackerModel.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def segment_from_bbox(
    image: str | Image.Image,
    bboxes: list[list[float]],
) -> list[dict]:
    """Produce a mask for each bounding box using SAM3 Tracker.

    Args:
        image: input image (path, URL, or PIL Image)
        bboxes: list of [x1, y1, x2, y2] boxes in pixel coordinates

    Returns a list (one per box) of dicts with:
        mask: (H, W) bool numpy array at original resolution
        iou_score: float — model-estimated mask quality
    """
    model, processor = get_or_load("sam3_tracker", _load)
    image = load_image(image)

    # Sam3TrackerProcessor expects [image][box][4]
    input_boxes = [[box] for box in bboxes]

    inputs = processor(
        images=image, input_boxes=[input_boxes], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]  # (num_boxes, num_candidates, H, W)

    iou_scores = outputs.iou_scores[0]  # (num_boxes, num_candidates)

    results = []
    for i in range(len(bboxes)):
        best_idx = iou_scores[i].argmax().item()
        results.append({
            "mask": masks[i, best_idx].numpy() > 0,
            "iou_score": round(iou_scores[i, best_idx].item(), 4),
        })
    return results
