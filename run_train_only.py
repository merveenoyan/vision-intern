"""Stage 3 only: train RF-DETR on the already-judged DocVQA dataset.

Judging (Stage 2) already produced ``merve/docvqa-media-judged`` (split:
``test``). RF-DETR support requires transformers>=5.10.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HF_TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

JUDGED_ID = "merve/docvqa-media-judged"

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
print("TRAINING COMPLETE")
print("=" * 60)
