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
"""HF Job — STAGE 2: score one judge over the moondream-labelled dataset.

Run once per judge (gemma-4-E4B-it and LFM2.5-VL-1.6B). Each writes its
per-detection verdicts to the run bucket; :mod:`jobs.merge_judges` combines them
into the ensemble ``judge_verdicts``. Two separate jobs sharing the bucket is
the intended multi-read/write pattern.

    # judge A (Google, ~8B)
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
      -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
      jobs/judge_one.py -- --model google/gemma-4-E4B-it \
      --out /data/docvqa-moondream/verdicts_gemma.parquet --max-samples 20

    # judge B (Liquid, 1.6B)
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
      -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
      jobs/judge_one.py -- --model LiquidAI/LFM2.5-VL-1.6B \
      --out /data/docvqa-moondream/verdicts_lfm.parquet --max-samples 20
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(REPO_DIR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Judge model id (HF repo)")
    p.add_argument("--dataset", default="merve/docvqa-media-labeled-moondream")
    p.add_argument("--split", default="test")
    p.add_argument("--detections-column", default="detections")
    p.add_argument("--overlay-column", default="detections_overlay",
                   help="Image column with numbered box overlays (rendered on "
                        "the fly when the column is absent).")
    p.add_argument("--out", required=True, help="Output verdicts parquet path")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    from datasets import load_dataset
    from PIL import Image

    from tools.utils import load_image
    from workflows.vlm_judge import score_detections

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Judging {len(ds)} rows with {args.model}")

    rows_out = []
    for i, row in enumerate(ds):
        dets = row.get(args.detections_column) or []
        if not dets:
            rows_out.append([])
            continue
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = load_image(img)
        overlay = row.get(args.overlay_column)
        verdicts = score_detections(
            img, dets, args.model,
            backend="transformers", base_url=None, api_key=None,
            overlay_img=overlay,
        )
        rows_out.append(verdicts)
        if i % 25 == 0:
            print(f"  [{i}/{len(ds)}] scored {len(verdicts)} dets")

    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_idx": list(range(len(rows_out))),
        "model": [args.model] * len(rows_out),
        "verdicts": [json.dumps(v) for v in rows_out],
    }).to_parquet(out)
    print(f"Wrote verdicts → {out}  (STAGE 2 judge DONE)")


if __name__ == "__main__":
    main()
