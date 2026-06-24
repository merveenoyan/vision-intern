# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "timm",
#   "datasets>=3.0",
#   "torchmetrics",
#   "pycocotools",
#   "pillow",
#   "numpy",
#   "huggingface_hub>=0.26",
#   "requests",
# ]
# ///
"""HF Job — evaluate a trained detector against the dataset's HUMAN ground truth.

The models were trained on VLM-*judged* labels, so their training ``eval_map``
measures agreement with the pipeline's own labels. This job instead scores a
model's predictions against the dataset's original human ``objects`` annotations
on a held-out split (default ``test``) — the true-quality number.

Class spaces are bridged by NAME: the GT ``ClassLabel`` index → its name → the
model's ``label2id``. GT boxes whose class the model can't predict are dropped
from the targets (reported), so the mAP isn't penalised for classes outside the
model's vocabulary.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN -e REPO_REF=multimodel-jobs -d \
      jobs/eval_vs_gt.py -- --model merve/rfdetr-roadsign-agree1 --split test
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
sys.path.insert(0, str(REPO_DIR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Trained detector repo id")
    p.add_argument("--dataset", default="Francesco/road-signs-6ih4y")
    p.add_argument("--dataset-config", default="default")
    p.add_argument("--split", default="test")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Score threshold for post-processing (0.0 for mAP).")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForObjectDetection.from_pretrained(args.model).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)

    # Model label space (lowercased name → model id).
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v.lower(): i for i, v in id2label.items()}
    print(f"Model has {len(id2label)} classes: {sorted(label2id)}", flush=True)

    ds = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # GT ClassLabel names (index → name), incl. the unused 'road-signs' supercat.
    # `objects` may be a dict-of-lists feature or a Sequence-of-dict — handle both.
    obj_feat = ds.features["objects"]
    cat_feat = (obj_feat["category"] if isinstance(obj_feat, dict)
                else obj_feat.feature["category"])
    gt_names = cat_feat.feature.names if hasattr(cat_feat, "feature") else cat_feat.names
    print(f"GT has {len(gt_names)} category names", flush=True)

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    n_gt_boxes = 0
    n_gt_dropped = 0  # GT boxes whose class the model can't predict
    n_pred_boxes = 0

    for i in range(len(ds)):
        row = ds[i]
        img = row["image"]
        if not isinstance(img, Image.Image):
            from tools.utils import load_image
            img = load_image(img)
        img = img.convert("RGB")
        w, h = img.size

        # --- prediction ---
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        post = processor.post_process_object_detection(
            out, threshold=args.threshold, target_sizes=[(h, w)])[0]
        pred = {
            "boxes": post["boxes"].cpu(),
            "scores": post["scores"].cpu(),
            "labels": post["labels"].cpu(),
        }
        n_pred_boxes += len(pred["labels"])

        # --- ground truth (COCO xywh → xyxy; GT class name → model id) ---
        obj = row["objects"]
        gboxes, glabels = [], []
        for bbox, cat in zip(obj["bbox"], obj["category"]):
            name = gt_names[int(cat)].lower()
            n_gt_boxes += 1
            if name not in label2id:
                n_gt_dropped += 1
                continue
            x, y, bw, bh = (float(v) for v in bbox)
            gboxes.append([x, y, x + bw, y + bh])
            glabels.append(label2id[name])
        target = {
            "boxes": torch.tensor(gboxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(glabels, dtype=torch.int64),
        }
        metric.update([pred], [target])
        if i % 50 == 0:
            print(f"  [{i}/{len(ds)}]", flush=True)

    res = metric.compute()
    print(f"\nPredicted {n_pred_boxes} boxes; GT {n_gt_boxes} boxes "
          f"({n_gt_dropped} dropped: class not in model)", flush=True)

    classes = res.pop("classes")
    map_pc = res.pop("map_per_class")
    mar_pc = res.pop("mar_100_per_class")

    print("\n=== Overall (vs HUMAN GT) ===", flush=True)
    for k in ("map", "map_50", "map_75", "map_small", "map_medium", "map_large",
              "mar_1", "mar_10", "mar_100"):
        if k in res:
            print(f"  {k:12s} {float(res[k]):.4f}", flush=True)

    print("\n=== Per-class mAP ===", flush=True)
    pairs = []
    cl = classes.tolist() if hasattr(classes, "tolist") else [classes]
    mp = map_pc.tolist() if hasattr(map_pc, "tolist") else [map_pc]
    mr = mar_pc.tolist() if hasattr(mar_pc, "tolist") else [mar_pc]
    for cid, cmap, cmar in zip(cl, mp, mr):
        pairs.append((id2label.get(int(cid), str(cid)), cmap, cmar))
    for name, cmap, cmar in sorted(pairs, key=lambda x: -x[1]):
        print(f"  {name:18s} map={cmap:.4f}  mar100={cmar:.4f}", flush=True)
    print("\nEVAL VS GT DONE", flush=True)


if __name__ == "__main__":
    main()
