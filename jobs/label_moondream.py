# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "accelerate",
#   "einops",
#   "pillow",
#   "datasets>=3.0",
#   "huggingface_hub>=0.26",
#   "pyarrow",
#   "pandas",
#   "requests",
# ]
# ///
"""HF Job — STAGE 1: label DocVQA media with moondream3 (native detection).

Loads moondream3, detects the media classes per image via ``.detect()``, pushes
a labelled dataset to the Hub **with a box-overlay gallery**, and records the
raw detections to the run bucket for the judge stage.

Run (smoke test, 20 imgs):
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
      -v hf://buckets/merve/vision-agent-runs:/data \
      -e REPO_REF=multimodel-jobs \
      jobs/label_moondream.py -- --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── bootstrap: clone repo for shared helpers ───────────────────────────
REPO_URL = os.environ.get("REPO_URL", "https://github.com/merveenoyan/vision-intern.git")
REPO_REF = os.environ.get("REPO_REF", "multimodel-jobs")
REPO_DIR = Path("/tmp/vision-intern")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                    REPO_URL, str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="lmms-lab/DocVQA")
    p.add_argument("--dataset-config", default="DocVQA")
    p.add_argument("--split", default="test")
    p.add_argument("--classes", default="table,image,chart,diagram,figure")
    p.add_argument("--model", default="moondream/moondream3-preview")
    p.add_argument("--output", default="merve/docvqa-media-labeled-moondream")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--bucket-dir", default="/data/docvqa-moondream")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset
    from PIL import Image

    from tools.hub_viz import push_dataset_with_viz
    from tools.moondream_detect import load_moondream, moondream_detect

    token = os.environ["HF_TOKEN"]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Loading {args.model} …")
    model = load_moondream(args.model, compile=not args.no_compile)

    ds = load_dataset(args.source, name=args.dataset_config, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Labelling {len(ds)} images with classes {classes}")

    all_dets: list[list[dict]] = []
    for i, row in enumerate(ds):
        img = row["image"]
        if not isinstance(img, Image.Image):
            from tools.utils import load_image
            img = load_image(img)
        dets = moondream_detect(img, classes, model)
        all_dets.append(dets)
        if i % 25 == 0:
            print(f"  [{i}/{len(ds)}] {len(dets)} dets")

    ds = ds.add_column("detections", all_dets)

    # Record raw detections to the bucket for the judge stage.
    bucket_dir = Path(args.bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({
        "row_idx": list(range(len(ds))),
        "detections": [json.dumps(d) for d in all_dets],
    }).to_parquet(bucket_dir / "detections.parquet")
    print(f"Wrote detections → {bucket_dir / 'detections.parquet'}")

    n_boxes = sum(len(d) for d in all_dets)
    print(f"Total {n_boxes} boxes over {len(ds)} images")

    push_dataset_with_viz(ds, args.output, token=token, image_column="image")
    print("STAGE 1 DONE")


if __name__ == "__main__":
    main()
