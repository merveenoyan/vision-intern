"""Eval a trained detector on a held-out split + dump GT-vs-pred overlays.

Loads a pushed ``AutoModelForObjectDetection`` checkpoint, runs it over the
given split, and (1) reports torchmetrics mAP/mAR against the dataset's
``detections`` pseudo-labels and (2) writes one side-by-side PNG per image
(left = ground-truth boxes, right = model predictions with confidence) into
``--out-dir`` for eyeballing. Runs locally on GPU if available.

    HF_TOKEN=$(hf auth token) python3 jobs/eval_overlay.py \
        --model merve/rfdetr-docvqa-media3-trainval-agree2-medium \
        --source merve/docvqa-media3-judged-splits-agree2 --split test \
        --out-dir viz_test_predictions/agree2-medium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Trained detector (HF repo)")
    p.add_argument("--source", required=True, help="Split dataset repo")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", required=True, help="Folder for overlay PNGs")
    p.add_argument("--metric-threshold", type=float, default=0.0,
                   help="Score floor for mAP (0.0 matches training eval).")
    p.add_argument("--viz-threshold", type=float, default=0.3,
                   help="Score floor for boxes drawn in the overlay panel.")
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap rendered overlays (mAP always uses all rows).")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    from tools.bbox_viz import draw_detections

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    print(f"Model classes: {id2label}", flush=True)

    ds = load_dataset(args.source, split=args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    n_viz = 0
    viz_cap = args.max_images if args.max_images is not None else len(ds)

    for i in range(len(ds)):
        row = ds[i]
        img = row["image"]
        if not isinstance(img, Image.Image):
            from tools.utils import load_image
            img = load_image(img)
        img = img.convert("RGB")
        w, h = img.size

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_object_detection(
            outputs, threshold=args.metric_threshold,
            target_sizes=torch.tensor([[h, w]]).to(device))[0]
        boxes = res["boxes"].cpu()
        scores = res["scores"].cpu()
        labels = res["labels"].cpu()

        # Ground truth from the pseudo-labels.
        gt = row["detections"] or []
        gt_boxes = torch.tensor([d["bbox"] for d in gt], dtype=torch.float32) \
            if gt else torch.zeros((0, 4))
        gt_labels = torch.tensor(
            [label2id.get(d["label"].lower(), -1) for d in gt], dtype=torch.long) \
            if gt else torch.zeros((0,), dtype=torch.long)

        metric.update(
            [{"boxes": boxes, "scores": scores, "labels": labels}],
            [{"boxes": gt_boxes, "labels": gt_labels}])

        if n_viz < viz_cap:
            gt_panel = draw_detections(img, gt)
            pred_dets = [
                {"bbox": [round(x) for x in boxes[j].tolist()],
                 "label": f"{id2label.get(labels[j].item(), labels[j].item())} "
                          f"{scores[j].item():.0%}"}
                for j in range(len(boxes)) if scores[j].item() >= args.viz_threshold
            ]
            pred_panel = draw_detections(img, pred_dets)
            combo = Image.new("RGB", (gt_panel.width + pred_panel.width,
                                      max(gt_panel.height, pred_panel.height)),
                              "white")
            combo.paste(gt_panel, (0, 0))
            combo.paste(pred_panel, (gt_panel.width, 0))
            combo.save(out_dir / f"{i:04d}_gt-vs-pred_{len(pred_dets)}preds.png")
            n_viz += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(ds)}] processed", flush=True)

    m = metric.compute()
    print(f"\n=== {args.model} on {args.source}[{args.split}] ===", flush=True)
    keys = ["map", "map_50", "map_75", "mar_100"]
    for k in keys:
        print(f"  {k:10s} {m[k].item():.4f}", flush=True)
    classes = m["classes"].tolist()
    for cid, cmap, cmar in zip(classes, m["map_per_class"].tolist(),
                               m["mar_100_per_class"].tolist()):
        print(f"  {id2label.get(cid, cid):10s} map={cmap:.4f}  mar100={cmar:.4f}",
              flush=True)
    print(f"Wrote {n_viz} overlays → {out_dir}")
    print("EVAL OVERLAY DONE", flush=True)


if __name__ == "__main__":
    main()
