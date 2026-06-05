"""Resume the DocVQA pipeline from Stage 2: judge existing labels → train.

Labeling (Stage 1) already produced ``merve/docvqa-media-labeled`` (split:
``test``), so this script skips it to avoid re-labeling / overwriting the Hub
dataset. It runs:

  - Stage 2 — Judge with Qwen3-VL-4B via local llama-server on :8084.
  - Stage 3 — Train RF-DETR on the judged dataset (test split).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HF_TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

JUDGE_MODEL = "Qwen3-VL-4B-Instruct-Q8_0.gguf"
JUDGE_BASE_URL = "http://localhost:8084/v1"
JUDGE_API_KEY = "sk-local-judge"

LABELED_ID = "merve/docvqa-media-labeled"
JUDGED_ID = "merve/docvqa-media-judged"

# ── Stage 2: Judge (Qwen3-VL-4B via local llama-server :8084) ────
print("\n" + "=" * 60)
print(f"STAGE 2 / 3 — Judging labels with {JUDGE_MODEL}")
print("=" * 60 + "\n", flush=True)

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
print("=" * 60 + "\n", flush=True)

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
print("PIPELINE COMPLETE (judge + train)")
print("=" * 60)
