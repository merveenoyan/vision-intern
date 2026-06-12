"""Visualize judged detections (with judge scores) on dataset images.

Reads the zero-confidence judged dataset and overlays each box's label and
the judge's score, so the (unreliable) scores are easy to eyeball. Saves to a
separate folder.
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from tools.bbox_viz import draw_detections


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize judged detections")
    p.add_argument("--dataset", default="merve/docvqa-media-judged")
    p.add_argument("--split", default="test")
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--output-dir", default="viz_judged")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    has_dets = [i for i in range(len(ds)) if ds[i]["detections"]]
    print(f"{len(has_dets)}/{len(ds)} rows have detections")

    random.seed(args.seed)
    indices = sorted(random.sample(has_dets, min(args.num, len(has_dets))))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for idx in indices:
        row = ds[idx]
        dets = row["detections"]
        verdicts = row.get("judge_verdicts")
        viz = draw_detections(row["image"], dets, verdicts)
        out_path = out_dir / f"judged_{idx:04d}.png"
        viz.save(out_path)
        labels = [d["label"] for d in dets]
        scores = [round(v.get("score", 0), 2) for v in (verdicts or [])]
        print(f"  [{idx}] {len(dets)} dets {labels} scores={scores} → {out_path}")

    print(f"\nSaved {len(indices)} images to {out_dir}/")


if __name__ == "__main__":
    main()
