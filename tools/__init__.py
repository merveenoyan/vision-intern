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
"""

from .detect import detect
from .instance_segment import instance_segment
from .segment_from_bbox import segment_from_bbox
from .segment_from_text import segment_from_text
from .depth import estimate_depth
from .pose import estimate_pose
from .grounded_detect import grounded_detect
from .fast_segment import fast_segment
from .ocr import ocr
from .vlm_detect import vlm_detect
from .document_ocr import document_ocr
from .ocr_judge import ocr_judge
from .bbox_utils import convert_bbox, convert_annotations, validate_annotations, compute_stats

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
]
