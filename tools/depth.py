"""Depth Anything 3 — monocular depth estimation."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from .utils import get_or_load, load_image

MODEL_ID = "depth-anything/DA3-LARGE-1.1"


def _load() -> tuple[Any, Any]:
    # DA3 ships its own library (`pip install depth-anything-3`); it does not
    # use transformers' AutoModelForDepthEstimation. There is no separate
    # processor, so the second slot of the (model, processor) cache is None.
    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(MODEL_ID).to(device=device)
    model.eval()
    return model, None


def estimate_depth(image: str | Image.Image) -> dict:
    """Estimate monocular depth for every pixel.

    Returns a dict with:
        depth_map: (H, W) float32 tensor at original resolution (higher = farther)
    """
    model, _ = get_or_load("depth_anything_3", _load)
    image = load_image(image)
    w, h = image.size

    # DA3's inference() takes a list of images (PIL/np/path); default export is
    # off (export_dir=None), and depth comes back at process_res (504), so we
    # interpolate to the original resolution below.
    prediction = model.inference([image])

    # prediction.depth is (N, H', W') float32; take the single frame.
    depth = prediction.depth[0]
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)
    depth = depth.float()

    # Resize back to the original image resolution.
    depth = F.interpolate(
        depth[None, None], size=(h, w), mode="bilinear", align_corners=False
    ).squeeze()

    return {"depth_map": depth.cpu()}
