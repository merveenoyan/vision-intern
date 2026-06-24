# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pillow",
#   "datasets>=3.0",
#   "huggingface_hub>=0.26",
#   "pyarrow",
#   "pandas",
#   "requests",
# ]
# ///
"""HF Job — STAGE 3: merge per-judge verdicts into the ensemble dataset.

CPU job. Reads the moondream-labelled dataset (images + detections) from the
Hub and each judge's verdict parquet from the run bucket, builds the ensemble
``judge_verdicts`` column (per-judge breakdown + vote), filters detections by
the ensemble policy, and pushes the judged dataset with a box-overlay gallery.

    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \
      -v hf://buckets/merve/vision-agent-runs:/data -e REPO_REF=multimodel-jobs \
      jobs/merge_judges.py -- \
      --verdicts "google/gemma-4-E4B-it::/data/docvqa-moondream/verdicts_gemma.parquet" \
      --verdicts "LiquidAI/LFM2.5-VL-1.6B::/data/docvqa-moondream/verdicts_lfm.parquet" \
      --min-agree 1 --max-samples 20
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
    p.add_argument("--dataset", default="merve/docvqa-media-labeled-moondream")
    p.add_argument("--split", default="test")
    p.add_argument("--detections-column", default="detections")
    p.add_argument("--output", default="merve/docvqa-media-judged-ensemble")
    p.add_argument("--verdicts", action="append", required=True,
                   help="Repeatable 'label::parquet_path' pair, one per judge.")
    p.add_argument("--min-agree", type=int, default=2,
                   help="Min judges voting 'correct' to keep a detection.")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--max-area-frac", type=float, default=0.9,
                   help="Drop detections whose box covers more than this "
                        "fraction of the page (non-VLM page-spanning guard).")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    import pandas as pd
    from datasets import load_dataset

    from tools.hub_viz import push_dataset_with_viz
    from workflows.vlm_judge import ensemble_row

    token = os.environ["HF_TOKEN"]

    # Parse "label::path" pairs and load each judge's per-row verdicts.
    judge_rows: dict[str, dict[int, list]] = {}
    for spec in args.verdicts:
        label, path = spec.split("::", 1)
        df = pd.read_parquet(path)
        judge_rows[label] = {
            int(r.row_idx): json.loads(r.verdicts) for r in df.itertuples()
        }
        print(f"Loaded {len(df)} rows of verdicts for '{label}' from {path}")
    labels = list(judge_rows)

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def _area_frac(det: dict, w: int, h: int) -> float:
        """Fraction of the page a detection's box covers (0 if unknown)."""
        bbox = det.get("bbox", det.get("box"))
        if not bbox or len(bbox) != 4 or w <= 0 or h <= 0:
            return 0.0
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1)) / float(w * h)

    filtered_dets: list[list[dict]] = []
    all_verdicts: list[list[dict]] = []
    dropped_geom = 0
    for i, row in enumerate(ds):
        dets = row.get(args.detections_column) or []
        if not dets:
            filtered_dets.append([])
            all_verdicts.append([])
            continue
        w, h = row["image"].size
        per_judge_row = {lbl: judge_rows[lbl].get(i, []) for lbl in labels}
        verdicts = ensemble_row(per_judge_row, len(dets), args.min_agree)
        kept = []
        for d, v in zip(dets, verdicts):
            # Record the geometric check on the verdict for transparency.
            v["area_frac"] = round(_area_frac(d, w, h), 4)
            v["geom_keep"] = v["area_frac"] <= args.max_area_frac
            if v["ensemble_keep"] and v["mean_score"] >= args.threshold:
                if v["geom_keep"]:
                    kept.append(d)
                else:
                    dropped_geom += 1
        filtered_dets.append(kept)
        all_verdicts.append(verdicts)

    total = sum(len(row.get(args.detections_column) or []) for row in ds)
    kept_total = sum(len(d) for d in filtered_dets)

    if args.detections_column in ds.column_names:
        ds = ds.remove_columns([args.detections_column])
    ds = ds.add_column(args.detections_column, filtered_dets)
    ds = ds.add_column("judge_verdicts", all_verdicts)

    print(f"Ensemble ({labels}, min_agree={args.min_agree}, "
          f"max_area_frac={args.max_area_frac}) kept {kept_total}/{total} "
          f"detections ({dropped_geom} dropped by page-spanning guard)")
    push_dataset_with_viz(ds, args.output, token=token, image_column="image")
    print("STAGE 3 DONE")


if __name__ == "__main__":
    main()
