"""Vision Agent Toolkit — vision tools with pluggable inference backends.

Tools that run VLMs (``vlm_detect``, ``document_ocr``, ``ocr_judge``)
support two backends via :mod:`tools.vlm_client`:

- ``backend="openai"`` — uses the OpenAI Python client, compatible with
  HF Inference Providers, vLLM, llama-server, and any OpenAI-compatible
  endpoint.
- ``backend="transformers"`` — loads the model locally on GPU.

Other tools use ``transformers`` directly and cache models in GPU memory.
All tools accept PIL Images (or paths/URLs) and return plain Python
dicts / lists.

Available tools
---------------
detect              RF-DETR closed-set detection (COCO 80)
instance_segment    RF-DETR-Seg instance segmentation (COCO 80)
segment_from_bbox   SAM3 Tracker — bbox prompts → high-quality masks
segment_from_text   Falcon-Perception — text prompt → zero-shot masks
estimate_depth      Depth Anything V2 — monocular relative depth
estimate_pose       Sapiens2 — dense 308-keypoint human pose
grounded_detect     MM-Grounding-DINO — open-vocabulary detection
fast_segment        EdgeTAM — lightweight bbox → mask (faster than SAM3)
ocr                 PaddleOCR-VL — vision-language OCR
vlm_detect          VLM instruction-prompted detection (free-form prompts)
document_ocr        Document OCR → markdown (configurable model + task modes)
ocr_judge           Pairwise OCR quality evaluation (VLM-as-judge + ELO)
convert_bbox        Convert bboxes between 6 formats (coco/xyxy/yolo/voc/tfod/ls)
validate_annotations  Validate detection annotations for common issues
compute_stats       Compute rich statistics for a COCO annotation file
dedupe_by_image     Collapse a dataset to one row per unique image
grouped_train_val_split  Train/val split with no image leaking across splits
image_key           Stable content hash of an image (for the two helpers above)
"""

import importlib

# Lazily map each public symbol → its submodule, so importing a light helper
# (e.g. ``tools.hub_viz``, ``tools.bbox_viz``) does not pull torch-heavy tools.
# ``from tools import detect`` still works — it triggers the import on access.
_LAZY = {
    "detect": "detect",
    "instance_segment": "instance_segment",
    "segment_from_bbox": "segment_from_bbox",
    "segment_from_text": "segment_from_text",
    "estimate_depth": "depth",
    "estimate_pose": "pose",
    "grounded_detect": "grounded_detect",
    "fast_segment": "fast_segment",
    "ocr": "ocr",
    "vlm_detect": "vlm_detect",
    "document_ocr": "document_ocr",
    "ocr_judge": "ocr_judge",
    "convert_bbox": "bbox_utils",
    "convert_annotations": "bbox_utils",
    "validate_annotations": "bbox_utils",
    "compute_stats": "bbox_utils",
    "dedupe_by_image": "dataset_utils",
    "grouped_train_val_split": "dataset_utils",
    "image_key": "dataset_utils",
}


def __getattr__(name: str):
    if name in _LAZY:
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "detect",
    "instance_segment",
    "segment_from_bbox",
    "segment_from_text",
    "estimate_depth",
    "estimate_pose",
    "grounded_detect",
    "fast_segment",
    "ocr",
    "vlm_detect",
    "document_ocr",
    "ocr_judge",
    "convert_bbox",
    "convert_annotations",
    "validate_annotations",
    "compute_stats",
    "dedupe_by_image",
    "grouped_train_val_split",
    "image_key",
]
