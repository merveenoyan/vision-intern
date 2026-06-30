"""Sapiens2 — human body-part semantic segmentation (28 parts + background)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .utils import get_or_load, load_image

# size -> checkpoint. All share the sapiens2 architecture, the
# Sapiens2ImageProcessor, and the AutoModelForSemanticSegmentation path;
# larger checkpoints trade speed for accuracy.
MODELS: dict[str, str] = {
    "0.4b": "facebook/sapiens2-seg-0.4b",
    "0.8b": "facebook/sapiens2-seg-0.8b",
    "1b": "facebook/sapiens2-seg-1b",
    "5b": "facebook/sapiens2-seg-5b",
}

# The 29-class body-part taxonomy (id -> name, RGB colour) the seg checkpoints
# were trained on. The model config only carries generic ``LABEL_n`` names, so
# the human-readable names + palette live here.
BODY_PARTS: list[tuple[str, tuple[int, int, int]]] = [
    ("Background", (50, 50, 50)), ("Apparel", (255, 218, 0)),
    ("Eyeglass", (14, 204, 182)), ("Face_Neck", (128, 200, 255)),
    ("Hair", (255, 0, 109)), ("Left_Foot", (189, 0, 204)),
    ("Left_Hand", (255, 0, 218)), ("Left_Lower_Arm", (0, 160, 204)),
    ("Left_Lower_Leg", (0, 255, 145)), ("Left_Shoe", (204, 0, 131)),
    ("Left_Sock", (182, 0, 255)), ("Left_Upper_Arm", (255, 109, 0)),
    ("Left_Upper_Leg", (0, 255, 255)), ("Lower_Clothing", (72, 0, 255)),
    ("Right_Foot", (204, 131, 0)), ("Right_Hand", (255, 0, 0)),
    ("Right_Lower_Arm", (72, 255, 0)), ("Right_Lower_Leg", (189, 204, 0)),
    ("Right_Shoe", (182, 255, 0)), ("Right_Sock", (102, 0, 204)),
    ("Right_Upper_Arm", (32, 72, 204)), ("Right_Upper_Leg", (0, 145, 255)),
    ("Torso", (14, 204, 0)), ("Upper_Clothing", (0, 128, 72)),
    ("Lower_Lip", (235, 205, 119)), ("Upper_Lip", (115, 227, 112)),
    ("Lower_Teeth", (157, 113, 143)), ("Upper_Teeth", (132, 93, 50)),
    ("Tongue", (82, 21, 114)),
]
PART_NAMES: list[str] = [name for name, _ in BODY_PARTS]
PALETTE: np.ndarray = np.array([color for _, color in BODY_PARTS], dtype=np.uint8)


def _load(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, processor


def segment_body_parts(
    image: str | Image.Image,
    size: str = "0.4b",
) -> dict:
    """Segment people into body parts (28 parts + background) with Sapiens2.

    Whole-image, human-centric semantic segmentation — no bounding boxes
    needed. Pixels with no person resolve to ``Background`` (id 0).

    Args:
        image: input image (path, URL, or PIL Image)
        size: which checkpoint — "0.4b" (default), "0.8b", "1b", or "5b"

    Returns a dict with:
        segmentation: (H, W) int32 numpy array — body-part class id per pixel
        segments_info: list of dicts (one per present part) with
            id (int), label (str), pixels (int)
        names: list[str] mapping class id -> body-part name (length 29)
    """
    if size not in MODELS:
        raise ValueError(f"size must be one of {sorted(MODELS)}, got {size!r}")

    model, processor = get_or_load(f"sapiens2_seg_{size}", lambda: _load(MODELS[size]))
    image = load_image(image)

    inputs = processor(image, return_tensors="pt").to(model.device)
    # Weights are bf16; cast only float tensors so integer inputs keep dtype.
    inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    seg = seg.cpu().numpy().astype(np.int32)

    segments_info = []
    for cid in np.unique(seg):
        cid = int(cid)
        name = PART_NAMES[cid] if cid < len(PART_NAMES) else f"LABEL_{cid}"
        segments_info.append({
            "id": cid,
            "label": name,
            "pixels": int((seg == cid).sum()),
        })

    return {
        "segmentation": seg,
        "segments_info": segments_info,
        "names": PART_NAMES,
    }
