"""Vision Agent Workflows — label, evaluate, and train detection models.

Pipeline
--------
1. **vlm_label**    Auto-annotate images using a VLM → COCO JSON
2. **vlm_judge**    Evaluate / filter annotations with a VLM-as-a-judge → cleaned COCO JSON
3. **train_rfdetr** Fine-tune RF-DETR on the curated dataset
"""

from .vlm_label import label_dataset
from .vlm_judge import judge_labels
from .train_rfdetr import train

__all__ = [
    "label_dataset",
    "judge_labels",
    "train",
]
