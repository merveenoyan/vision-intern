"""Vision Agent Workflows — label, evaluate, and train detection models.

Pipeline
--------
1. **vlm_label**       Auto-annotate images using a VLM → COCO JSON
2. **gen_descriptions** Write {label: definition} judge descriptions for human review
3. **vlm_judge**       Evaluate / filter annotations with a VLM-as-a-judge → cleaned COCO JSON
4. **train_rfdetr**    Fine-tune RF-DETR on the curated dataset
"""

import importlib

# Lazy so importing a light entry point (e.g. ``vlm_judge.ensemble_row`` in a
# CPU-only merge job) does not pull torch via ``train_rfdetr``.
_LAZY = {
    "label_dataset": "vlm_label",
    "judge_labels": "vlm_judge",
    "gen_descriptions": "gen_descriptions",
    "train": "train_rfdetr",
}


def __getattr__(name: str):
    if name in _LAZY:
        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "label_dataset",
    "judge_labels",
    "gen_descriptions",
    "train",
]
