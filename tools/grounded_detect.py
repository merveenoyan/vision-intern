"""MM-Grounding-DINO — open-vocabulary object detection from text queries."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .utils import get_or_load, load_image

# size -> checkpoint. Both share the mm-grounding-dino architecture, processor,
# and inference path; "large" trades speed for accuracy.
MODELS: dict[str, str] = {
    "tiny": "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
    "large": "openmmlab-community/mm_grounding_dino_large_all",
}


def _load(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    # float32: the deformable-attention grid_sample op has no bf16 kernel.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float32
    )
    model.eval()
    return model, processor


def grounded_detect(
    image: str | Image.Image,
    text_queries: list[str],
    threshold: float = 0.3,
    text_threshold: float = 0.25,
    size: str = "tiny",
) -> list[dict]:
    """Detect objects matching free-form *text_queries*.

    Args:
        image: input image (path, URL, or PIL Image)
        text_queries: open-vocabulary class names, e.g. ["cat", "remote control"]
        threshold: detection confidence threshold
        text_threshold: per-token threshold for phrase extraction
        size: which checkpoint to use — "tiny" (fast) or "large" (more accurate)

    Returns a list of detections, each a dict with:
        label (str): matched text phrase
        score (float): confidence
        box (list[float]): [x1, y1, x2, y2] in pixel coordinates
    """
    if size not in MODELS:
        raise ValueError(f"size must be one of {sorted(MODELS)}, got {size!r}")

    model_id = MODELS[size]
    model, processor = get_or_load(f"mm_grounding_dino_{size}", lambda: _load(model_id))
    image = load_image(image)

    inputs = processor(
        images=image, text=[text_queries], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        detections.append({
            "label": label,
            "score": round(score.item(), 4),
            "box": [round(c, 1) for c in box.tolist()],
        })
    return detections
