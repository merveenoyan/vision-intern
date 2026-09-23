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
(``workflows.train_rfdetr``) on an explicitly selected ensemble-policy dataset,
holding out 15% for mAP, and pushes the model to the Hub. Pipeline detections
are selected explicitly so an inherited ``objects`` column cannot shadow them.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN --timeout 6h \
      -e REPO_REF=multimodel-jobs \
      jobs/train_rfdetr_job.py -- \
      --source merve/docvqa-media-judged-ensemble-agree1 \
      --hub-model-id merve/rfdetr-docvqa-qwen-agree1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/merveenoyan/vision-intern.git")
REPO_REF = os.environ.get("REPO_REF", "multimodel-jobs")
# Set REPO_DIR to a local checkout (e.g. `REPO_DIR=$(pwd)`) to run against your
# working tree; left unset it clones REPO_REF (the default on HF Jobs).
REPO_DIR = Path(os.environ.get("REPO_DIR", "/tmp/vision-intern"))
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                    REPO_URL, str(REPO_DIR)], check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--train-split", default="test")
    p.add_argument("--val-split", default="none")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--model", default="Roboflow/rf-detr-large",
                   help="Base checkpoint, or a prior fine-tune to CONTINUE from.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Peak LR. Lower it (e.g. 1e-5) when continuing from an "
                        "already-converged checkpoint.")
    p.add_argument("--annotation-source", default="detections",
                   choices=["auto", "objects", "detections"],
                   help="Annotation column passed to the trainer.")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable Albumentations aug. Needed for direction/colour"
                        "-coded classes: HorizontalFlip mirrors left/right signs "
                        "and colour jitter scrambles light colours, both "
                        "label-breaking for road signs.")
    p.add_argument("--output-dir", default="checkpoints/rfdetr-finetuned")
    p.add_argument("--hub-model-id", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()

    cmd = [
        sys.executable, "-m", "workflows.train_rfdetr",
        "--source", args.source,
        "--train-split", args.train_split,
        "--val-split", args.val_split,
        "--val-size", str(args.val_size),
        "--annotation-source", args.annotation_source,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        *(["--no-augment"] if args.no_augment else []),
        "--output-dir", args.output_dir,
        "--push-to-hub",
        "--hub-model-id", args.hub_model_id,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR))
    print("STAGE 4 DONE")


if __name__ == "__main__":
    main()
