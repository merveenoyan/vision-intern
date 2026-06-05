"""EdgeTAM — lightweight, fast mask generation from bounding boxes."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "yonigozlan/EdgeTAM-hf"


def _load() -> tuple[Any, Any]:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def fast_segment(
    image: str | Image.Image,
    bboxes: list[list[float]],
) -> list[dict]:
    """Generate masks from bounding boxes using EdgeTAM (faster than SAM3).

    Args:
        image: input image (path, URL, or PIL Image)
        bboxes: list of [x1, y1, x2, y2] boxes in pixel coordinates

    Returns a list (one per box) of dicts with:
        mask: (H, W) bool numpy array at original resolution
        iou_score: float — model-estimated mask quality
    """
    model, processor = get_or_load("edgetam", _load)
    image = load_image(image)

    # Sam2Processor expects input_boxes as [image][box][4]
    inputs = processor(
        images=image, input_boxes=[bboxes], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], binarize=True
    )[0]  # (num_boxes, num_candidates, H, W)

    iou_scores = outputs.iou_scores[0]  # (num_boxes, num_candidates)

    results = []
    for i in range(len(bboxes)):
        best_idx = iou_scores[i].argmax().item()
        results.append({
            "mask": masks[i, best_idx].cpu().numpy() > 0,
            "iou_score": round(iou_scores[i, best_idx].item(), 4),
        })
    return results
