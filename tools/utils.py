from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PIL import Image

if TYPE_CHECKING:
    import numpy as np
    import torch

_registry_lock = threading.Lock()
_model_cache: dict[str, tuple[Any, Any]] = {}


def get_or_load(name: str, load_fn: Callable[[], tuple[Any, Any]]) -> tuple[Any, Any]:
    """Return cached (model, processor) or call *load_fn* once to populate the cache."""
    if name not in _model_cache:
        with _registry_lock:
            if name not in _model_cache:
                _model_cache[name] = load_fn()
    return _model_cache[name]


def load_image(source: str | Path | Image.Image) -> Image.Image:
    """Accept a file path, URL, or PIL Image and return an RGB PIL Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    source = str(source)
    if source.startswith(("http://", "https://")):
        import requests
        return Image.open(requests.get(source, stream=True, timeout=30).raw).convert("RGB")
    return Image.open(source).convert("RGB")


def masks_to_rle(masks: "np.ndarray | torch.Tensor") -> list[dict]:
    """Encode binary masks (N, H, W) to COCO-style RLE dicts via pycocotools."""
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    masks = np.asfortranarray(masks.astype(np.uint8))
    rles = []
    for m in masks:
        rle = mask_utils.encode(m)
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode a single COCO RLE dict to a binary (H, W) bool array."""
    from pycocotools import mask as mask_utils

    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return mask_utils.decode({"size": rle["size"], "counts": counts}).astype(bool)
