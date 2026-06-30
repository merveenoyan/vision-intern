"""Sapiens2 — human surface normal estimation."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

# size -> checkpoint. All share the sapiens2 architecture and the
# AutoModelForNormalEstimation path; larger checkpoints trade speed for accuracy.
MODELS: dict[str, str] = {
    "0.4b": "facebook/sapiens2-normal-0.4b",
    "0.8b": "facebook/sapiens2-normal-0.8b",
    "1b": "facebook/sapiens2-normal-1b",
    "5b": "facebook/sapiens2-normal-5b",
}


def _load(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForNormalEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForNormalEstimation.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def estimate_surface_normals(
    image: str | Image.Image,
    size: str = "0.4b",
) -> dict:
    """Estimate per-pixel surface normals on human-centric imagery with Sapiens2.

    Args:
        image: input image (path, URL, or PIL Image)
        size: which checkpoint — "0.4b" (default), "0.8b", "1b", or "5b"

    Returns a dict with:
        normals: (H, W, 3) float32 numpy array — L2-normalized XYZ unit
            vectors in [-1, 1] at the original image resolution. Visualize as
            RGB with ``((normals + 1) / 2 * 255)``.
    """
    if size not in MODELS:
        raise ValueError(f"size must be one of {sorted(MODELS)}, got {size!r}")

    model, processor = get_or_load(f"sapiens2_normal_{size}", lambda: _load(MODELS[size]))
    image = load_image(image)

    inputs = processor(image, return_tensors="pt").to(model.device)
    # Weights are bf16; cast only float tensors so integer inputs keep dtype.
    inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    original_size = (image.height, image.width)
    result = processor.post_process_normal_estimation(
        outputs, source_sizes=[original_size], target_sizes=[original_size]
    )[0]

    # (3, H, W) float -> (H, W, 3) float32 on CPU.
    normals = result["normals"].float().permute(1, 2, 0).cpu().numpy()
    return {"normals": normals}
