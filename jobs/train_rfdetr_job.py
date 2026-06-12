# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "timm",
#   "albumentations",
#   "accelerate",
#   "datasets>=3.0",
#   "torchmetrics",
#   "pycocotools",
#   "huggingface_hub>=0.26",
#   "pillow",
#   "numpy",
#   "requests",
# ]
# ///
"""HF Job — STAGE 4: train RF-DETR on the ensemble-judged dataset.

Clones the repo and runs the existing generalized trainer
(``workflows.train_rfdetr``) on ``merve/docvqa-media-judged-ensemble``, holding
out 15% for mAP, and pushes the model to the Hub.

    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 6h \
      -e REPO_REF=multimodel-jobs \
      jobs/train_rfdetr_job.py -- --epochs 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/merveenoyan/vision-intern.git")
REPO_REF = os.environ.get("REPO_REF", "multimodel-jobs")
REPO_DIR = Path("/tmp/vision-intern")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                    REPO_URL, str(REPO_DIR)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="merve/docvqa-media-judged-ensemble")
    p.add_argument("--train-split", default="test")
    p.add_argument("--val-split", default="none")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--model", default="Roboflow/rf-detr-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", default="checkpoints/rfdetr-docvqa-moondream")
    p.add_argument("--hub-model-id", default="merve/rfdetr-docvqa-moondream")
    args = p.parse_args()

    cmd = [
        sys.executable, "-m", "workflows.train_rfdetr",
        "--source", args.source,
        "--train-split", args.train_split,
        "--val-split", args.val_split,
        "--val-size", str(args.val_size),
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--output-dir", args.output_dir,
        "--push-to-hub",
        "--hub-model-id", args.hub_model_id,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR))
    print("STAGE 4 DONE")


if __name__ == "__main__":
    main()
