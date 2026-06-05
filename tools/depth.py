"""Depth Anything V2 — monocular relative depth estimation."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"


def _load() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float32
    )
    model.eval()
    return model, processor


def estimate_depth(image: str | Image.Image) -> dict:
    """Estimate relative depth for every pixel.

    Returns a dict with:
        depth_map: (H, W) float32 tensor at original resolution (higher = farther)
    """
    model, processor = get_or_load("depth_anything_v2", _load)
    image = load_image(image)
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    depth = outputs.predicted_depth  # (1, h', w')
    depth = F.interpolate(
        depth.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze()

    return {"depth_map": depth.cpu()}
