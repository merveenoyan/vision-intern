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
"""HF Job — STAGE 2: score one judge over the labelled dataset (GPU, batched).

Run once per judge (gemma-4-E4B-it and LFM2.5-VL-1.6B). These judges are small
and **not served by HF Inference Providers**, so they run **locally with the
``transformers`` backend on the job's GPU** — not through the router. Inference
is **batched** (``--batch-size``, many images per ``generate`` call) so the GPU
is not starved one image at a time.

Each writes its per-detection verdicts to the run bucket; :mod:`jobs.merge_judges`
combines them into the ensemble ``judge_verdicts``. Two separate jobs sharing the
bucket is the intended multi-read/write pattern.

    # judge A (Google, ~8B) — the long pole, so it gets the big GPU
    hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN --timeout 3h \
      -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
      jobs/judge_one.py -- --model google/gemma-4-E4B-it \
      --out /data/run/verdicts_gemma.parquet --batch-size 8

    # judge B (Liquid, 1.6B) — small, fits l4x1
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN --timeout 3h \
      -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
      jobs/judge_one.py -- --model LiquidAI/LFM2.5-VL-1.6B \
      --out /data/run/verdicts_lfm.parquet --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Source of tools/ + workflows/. Set REPO_DIR to a local checkout (e.g.
# `REPO_DIR=$(pwd)`) to run against your working tree; left unset it clones
# REPO_REF (the default on HF Jobs, which has no checkout).
REPO_URL = os.environ.get("REPO_URL", "https://github.com/merveenoyan/vision-intern.git")
REPO_REF = os.environ.get("REPO_REF", "multimodel-jobs")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/tmp/vision-intern"))
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
    p.add_argument("--out", required=True,
                   help="Verdicts parquet: a relative name is placed under the "
                        "data root (--data-root / $DATA_ROOT / /data mount / the "
                        "bucket); an absolute path or hf:// URI is used as-is.")
    p.add_argument("--data-root", default=None,
                   help="Where run artifacts live (local dir, /data, or "
                        "hf://buckets/<id>). Default: auto (mount if present, "
                        "else the bucket over hf://).")
    p.add_argument("--bucket", default=None,
                   help="Bucket id for the hf:// fallback (default "
                        "merve/vision-agent-runs).")
    p.add_argument("--class-descriptions", default=None,
                   help="JSON {label: definition} the human approved (local "
                        "path or hf:// URI). Injected into the judge prompt so "
                        "it evaluates against the agreed definitions.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Images per GPU batch. These judges are not on HF "
                        "Inference Providers, so they run locally with "
                        "transformers — batching many images per generate() "
                        "call is the throughput lever. Lower it if you OOM.")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    from tools import run_store

    from datasets import load_dataset
    from PIL import Image

    from tools.utils import load_image
    from workflows.vlm_judge import score_detections_batch

    # Load the human-approved category definitions (local or hf://).
    class_descriptions = None
    if args.class_descriptions:
        spec = args.class_descriptions
        if "://" in spec:
            import fsspec
            with fsspec.open(spec, "r") as f:
                raw = f.read()
        else:
            raw = Path(spec).read_text()
        class_descriptions = {str(k).lower(): str(v) for k, v in json.loads(raw).items()}
        print(f"Loaded {len(class_descriptions)} approved definition(s) from {spec}")

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Judging {len(ds)} rows with {args.model} "
          f"(batch_size={args.batch_size})")

    # Assemble (img, detections, overlay) per row, then score in GPU batches.
    items = []
    for row in ds:
        dets = row.get(args.detections_column) or []
        if not dets:
            items.append((None, [], None))
            continue
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = load_image(img)
        items.append((img, dets, row.get(args.overlay_column)))

    rows_out = score_detections_batch(
        items, args.model,
        backend="transformers", base_url=None, api_key=None,
        class_descriptions=class_descriptions,
        batch_size=args.batch_size,
    )
    print(f"  scored {sum(len(v) for v in rows_out)} detections "
          f"across {len(rows_out)} rows")

    import pandas as pd
    out = run_store.resolve_artifact(
        args.out, data_root=args.data_root,
        bucket=args.bucket or run_store.DEFAULT_BUCKET,
    )
    run_store.write_parquet(pd.DataFrame({
        "row_idx": list(range(len(rows_out))),
        "model": [args.model] * len(rows_out),
        "verdicts": [json.dumps(v) for v in rows_out],
    }), out)
    print(f"Wrote verdicts → {out}  (STAGE 2 judge DONE)")


if __name__ == "__main__":
    main()
