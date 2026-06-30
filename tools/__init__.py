"""Vision Agent Toolkit — vision tools with pluggable inference backends.

Tools that run VLMs (``vlm_detect``, ``ocr_judge``) support two backends
via :mod:`tools.vlm_client`:

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
estimate_depth      Depth Anything 3 — monocular depth (DA3-LARGE)
estimate_pose       Sapiens2 — dense 308-keypoint human pose
segment_body_parts  Sapiens2 — human body-part segmentation (28 parts + bg)
estimate_surface_normals  Sapiens2 — human surface normal estimation
matte_human         Sapiens2 — human image matting (alpha + foreground)
grounded_detect     MM-Grounding-DINO — open-vocab detection (size: tiny/large)
fast_segment        EdgeTAM — lightweight bbox → mask (faster than SAM3)
ocr                 Vision-language OCR (size: large/medium/small)
vlm_detect          VLM instruction-prompted detection (free-form prompts)
ocr_judge           Pairwise OCR quality evaluation (VLM-as-judge + ELO)
convert_bbox        Convert bboxes between 6 formats (coco/xyxy/yolo/voc/tfod/ls)
validate_annotations  Validate detection annotations for common issues
compute_stats       Compute rich statistics for a COCO annotation file
annotate            supervision-backed bbox visualization (per-class/track colours)
to_supervision / from_supervision  Detection-dicts ↔ supervision.Detections
track_video         Roboflow trackers + supervision tracking visualization (video)
dedupe_by_image     Collapse a dataset to one row per unique image
grouped_train_val_split  Train/val split with no image leaking across splits
image_key           Stable content hash of an image (for the two helpers above)

Agent tool layer
----------------
get_tools / get_tool / list_tools   Discover registered tools (+ JSON Schema)
as_json_schema      Framework-agnostic ``{name, description, parameters}`` specs
call                Invoke a tool by name, injecting hidden worker config
configure / ToolConfig   Set per-role (default/labeller/judge) endpoint config
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
    "segment_body_parts": "segment_body_parts",
    "estimate_surface_normals": "surface_normals",
    "matte_human": "human_matting",
    "grounded_detect": "grounded_detect",
    "fast_segment": "fast_segment",
    "ocr": "ocr",
    "vlm_detect": "vlm_detect",
    "ocr_judge": "ocr_judge",
    "convert_bbox": "bbox_utils",
    "convert_annotations": "bbox_utils",
    "validate_annotations": "bbox_utils",
    "compute_stats": "bbox_utils",
    # supervision-backed visualization + Roboflow tracking (the `viz` extra;
    # supervision imported lazily inside the functions).
    "annotate": "sv_viz",
    "to_supervision": "sv_convert",
    "from_supervision": "sv_convert",
    "track_video": "track_video",
    "dedupe_by_image": "dataset_utils",
    "grouped_train_val_split": "dataset_utils",
    "image_key": "dataset_utils",
    # Agent tool layer (registry + config are torch-free).
    "get_tools": "registry",
    "get_tool": "registry",
    "list_tools": "registry",
    "as_json_schema": "registry",
    "call": "registry",
    "ToolSpec": "registry",
    "configure": "config",
    "ToolConfig": "config",
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
    "segment_body_parts",
    "estimate_surface_normals",
    "matte_human",
    "grounded_detect",
    "fast_segment",
    "ocr",
    "vlm_detect",
    "ocr_judge",
    "convert_bbox",
    "convert_annotations",
    "validate_annotations",
    "compute_stats",
    "annotate",
    "to_supervision",
    "from_supervision",
    "track_video",
    "dedupe_by_image",
    "grouped_train_val_split",
    "image_key",
    "get_tools",
    "get_tool",
    "list_tools",
    "as_json_schema",
    "call",
    "ToolSpec",
    "configure",
    "ToolConfig",
]
