"""Sapiens2 — human image matting (alpha + foreground extraction)."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

# Matting ships a single checkpoint (the 1B variant).
MODEL_ID = "facebook/sapiens2-matting-1b"


def _load() -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForImageMatting

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageMatting.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def matte_human(
    image: str | Image.Image,
    background: list[int] | None = None,
) -> dict:
    """Extract a soft alpha matte and foreground for the person(s) in *image*.

    Args:
        image: input image (path, URL, or PIL Image)
        background: optional [r, g, b] (0-255) solid colour to composite the
            foreground over (e.g. [0, 177, 64] for chroma green). When given,
            a ``composite`` image is returned.

    Returns a dict with:
        alpha: (H, W) float32 numpy array in [0, 1] — per-pixel opacity
        foreground: (H, W, 3) float32 numpy array in [0, 1] — estimated
            foreground colours
        composite: (H, W, 3) uint8 numpy array — only present when *background*
            is given; the foreground composited over the solid background
    """
    model, processor = get_or_load("sapiens2_matting", _load)
    image = load_image(image)

    inputs = processor(image, return_tensors="pt").to(model.device)
    # Weights are bf16; cast only float tensors so integer inputs keep dtype.
    inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    original_size = (image.height, image.width)
    backgrounds = None
    if background is not None:
        backgrounds = torch.tensor(background, dtype=torch.uint8).view(3, 1, 1)

    result = processor.post_process_image_matting(
        outputs, target_sizes=[original_size], backgrounds=backgrounds
    )[0]

    out = {
        # alpha: (1, H, W) -> (H, W)
        "alpha": result["alpha"].squeeze(0).float().cpu().numpy(),
        # foreground: (3, H, W) -> (H, W, 3)
        "foreground": result["foreground"].float().permute(1, 2, 0).cpu().numpy(),
    }
    if "composite" in result:
        out["composite"] = result["composite"].permute(1, 2, 0).cpu().numpy()
    return out
