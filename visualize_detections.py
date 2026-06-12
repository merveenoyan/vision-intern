"""Visualize bounding box detections on dataset images."""

import random
from pathlib import Path

from datasets import load_dataset

from tools.bbox_viz import draw_detections

DATASET_ID = "merve/docvqa-media-labeled"
SPLIT = "test"
NUM_EXAMPLES = 8
OUTPUT_DIR = Path("viz_output")


def main() -> None:
    ds = load_dataset(DATASET_ID, split=SPLIT)

    has_dets = [i for i in range(len(ds)) if ds[i]["detections"]]
    print(f"{len(has_dets)}/{len(ds)} rows have detections")

    indices = random.sample(has_dets, min(NUM_EXAMPLES, len(has_dets)))
    indices.sort()

    OUTPUT_DIR.mkdir(exist_ok=True)

    for idx in indices:
        row = ds[idx]
        img = row["image"]
        dets = row["detections"]

        viz = draw_detections(img, dets)

        out_path = OUTPUT_DIR / f"example_{idx:04d}.png"
        viz.save(out_path)

        labels = [d["label"] for d in dets]
        print(f"  [{idx}] {len(dets)} detections {labels} → {out_path}")

    print(f"\nSaved {len(indices)} images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
