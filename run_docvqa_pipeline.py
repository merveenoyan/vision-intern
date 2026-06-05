"""End-to-end pipeline: label DocVQA docs → judge → push to Hub → train RF-DETR.

Model roles (see README "Recommended architecture"):
  - Labeller (~7-8B): Qwen3-VL-8B via HF Inference Providers (remote).
  - Judge (~4B):      Qwen3-VL-4B via local llama-server on :8084.
  - Orchestrator (~12B, Gemma-4) drives/babysits this run and never
    labels or judges itself.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HF_TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

DATASET = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"
CLASSES = ["table", "image", "chart", "diagram", "figure"]

LABEL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
JUDGE_MODEL = "Qwen3-VL-4B-Instruct-Q8_0.gguf"
JUDGE_BASE_URL = "http://localhost:8084/v1"
JUDGE_API_KEY = "sk-local-judge"

LABELED_ID = "merve/docvqa-media-labeled"
JUDGED_ID = "merve/docvqa-media-judged"
MAX_SAMPLES = 1000

# ── Stage 1: Label (Qwen3-VL-8B via HF Inference Providers) ─────
print("\n" + "=" * 60)
print(f"STAGE 1 / 3 — Labeling {MAX_SAMPLES} images with {LABEL_MODEL}")
print("=" * 60 + "\n")

from workflows.vlm_label import label_dataset

label_dataset(
    source=DATASET,
    classes=CLASSES,
    output=LABELED_ID,
    model_id=LABEL_MODEL,
    backend="openai",
    api_key=HF_TOKEN,
    image_column="image",
    split="test",
    max_samples=MAX_SAMPLES,
    push_to_hub=True,
    hf_token=HF_TOKEN,
    dataset_config=DATASET_CONFIG,
)

# ── Stage 2: Judge (Qwen3-VL-4B via local llama-server :8084) ────
print("\n" + "=" * 60)
print(f"STAGE 2 / 3 — Judging labels with {JUDGE_MODEL}")
print("=" * 60 + "\n")

from workflows.vlm_judge import judge_labels

judge_labels(
    source=LABELED_ID,
    output=JUDGED_ID,
    model_id=JUDGE_MODEL,
    threshold=0.5,
    backend="openai",
    base_url=JUDGE_BASE_URL,
    api_key=JUDGE_API_KEY,
    image_column="image",
    detections_column="detections",
    split="test",
    push_to_hub=True,
    hf_token=HF_TOKEN,
)

# ── Stage 3: Train RF-DETR ──────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3 / 3 — Training RF-DETR (10 epochs)")
print("=" * 60 + "\n")

from workflows.train_rfdetr import train

train(
    source=JUDGED_ID,
    output_dir="checkpoints/rfdetr-docvqa",
    epochs=10,
    batch_size=4,
    train_split="test",
    val_split=None,
)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
