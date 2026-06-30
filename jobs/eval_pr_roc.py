# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers",
#   "timm",
#   "datasets>=3.0",
#   "scikit-learn",
#   "matplotlib",
#   "pillow",
#   "numpy",
#   "huggingface_hub>=0.26",
#   "requests",
# ]
# ///
"""HF Job — precision / recall / ROC for detector(s) vs HUMAN ground truth.

Companion to ``jobs/eval_vs_gt.py`` (which reports COCO mAP/mAR). This job
frames detection as a **per-prediction binary decision** so it can report the
metrics the user asked for: precision, recall, F1 and an ROC curve, scored
against the dataset's original human ``objects`` annotations on a held-out
split (default ``test``).

Matching (per image, per class, greedy by descending score, IoU >= --iou):
each predicted box is a **true positive** if it lands on an unmatched GT box of
the same class, else a **false positive**; GT boxes never matched are **false
negatives**. Class spaces are bridged by NAME (GT ``ClassLabel`` index → name →
model ``label2id``); GT boxes whose class the model can't predict are dropped
from the targets (reported), so metrics aren't penalised for out-of-vocab
classes.

Curves:
  * **PR curve / AP** — precision vs recall as the score threshold sweeps down;
    recall denominator is the GT count. This is the detection-native curve.
  * **ROC curve / AUC** — each prediction is one sample whose label is TP(1)/
    FP(0) and whose score is the model confidence; ROC plots TPR vs FPR over
    *predictions*. Detection has no natural true-negative pool, so this measures
    **how well the confidence score separates real detections from false ones**
    (confidence calibration), NOT 1-specificity over all possible boxes. Read it
    that way; the PR curve and mAP are the box-quality numbers.

Operating-point precision/recall/F1 are reported at --op-threshold (default 0.5).

Evaluates every model in --models and overlays them on shared PR + ROC plots,
then uploads the plots, a per-class table and ``metrics.json`` to a Hub dataset
repo (--output-repo).

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN -e REPO_REF=roadsign-e2e -d \
      jobs/eval_pr_roc.py -- \
      --models merve/rfdetr-roadsign-agree1-large-noaug,merve/rfdetr-roadsign-agree2-large-noaug \
      --dataset Francesco/road-signs-6ih4y --split test \
      --output-repo merve/rfdetr-roadsign-pr-roc
"""

from __future__ import annotations

import argparse
import io
import json
import os


def _iou_matrix(pred_boxes, gt_boxes):
    """Vectorised IoU between two sets of xyxy boxes -> (n_pred, n_gt)."""
    import numpy as np

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    p = np.asarray(pred_boxes, dtype=np.float32)[:, None, :]   # (P,1,4)
    g = np.asarray(gt_boxes, dtype=np.float32)[None, :, :]     # (1,G,4)
    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = ap + ag - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_image(pred_boxes, pred_scores, pred_labels,
                 gt_boxes, gt_labels, iou_thr):
    """Greedy match one image. Returns (records, per_class_fn).

    ``records`` is a list of (score, is_tp, label) — one per prediction.
    ``per_class_fn`` maps class id -> count of unmatched GT boxes (false negs).
    """
    import numpy as np

    order = np.argsort(-np.asarray(pred_scores)) if len(pred_scores) else []
    ious = _iou_matrix(pred_boxes, gt_boxes)
    gt_used = [False] * len(gt_labels)
    records = []
    for pi in order:
        plabel = pred_labels[pi]
        best_j, best_iou = -1, iou_thr
        for gj in range(len(gt_labels)):
            if gt_used[gj] or gt_labels[gj] != plabel:
                continue
            if ious[pi, gj] >= best_iou:
                best_iou = ious[pi, gj]
                best_j = gj
        is_tp = best_j >= 0
        if is_tp:
            gt_used[best_j] = True
        records.append((float(pred_scores[pi]), bool(is_tp), int(plabel)))

    per_class_fn = {}
    for gj, used in enumerate(gt_used):
        if not used:
            per_class_fn[gt_labels[gj]] = per_class_fn.get(gt_labels[gj], 0) + 1
    return records, per_class_fn


def evaluate_model(model_id, ds, gt_names, iou_thr, op_threshold, score_floor,
                   aliases=None, exclude=None):
    """Run one model over the dataset and compute curves + operating point.

    ``aliases`` maps a name -> canonical name, applied to BOTH the model's
    label space and the GT names so divergently-named-but-identical classes
    (e.g. model's ``ped_crossing`` vs GT's ``ped_zebra_cross``) match during
    scoring. ``exclude`` is a set of canonical names dropped from both sides.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    exclude = {e.lower() for e in (exclude or set())}

    def canon(name):
        return aliases.get(name.lower(), name.lower())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)

    raw_id2label = {int(k): v for k, v in model.config.id2label.items()}
    # Canonical class space = the model's producible names after aliasing,
    # minus excluded names. Both predictions and GT are mapped into it.
    canon_names = sorted({canon(v) for v in raw_id2label.values()} - exclude)
    name2cid = {n: i for i, n in enumerate(canon_names)}
    id2label = {i: n for n, i in name2cid.items()}      # canonical id -> name
    # model raw label id -> canonical id (None if the canon name is excluded)
    raw2cid = {k: name2cid.get(canon(v)) for k, v in raw_id2label.items()}
    label2id = name2cid                                  # canonical name -> id
    print(f"[{model_id}] {len(name2cid)} classes: {sorted(name2cid)}", flush=True)
    if aliases:
        print(f"[{model_id}] aliases applied: {aliases}", flush=True)
    if exclude:
        print(f"[{model_id}] excluded: {sorted(exclude)}", flush=True)

    records = []                      # (score, is_tp, label) across all images
    fn_per_class = {}                 # class id -> false negatives
    gt_per_class = {}                 # class id -> total GT boxes (after bridge)
    n_gt_boxes = n_gt_dropped = n_pred = 0

    for i in range(len(ds)):
        row = ds[i]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        w, h = img.size

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        post = processor.post_process_object_detection(
            out, threshold=score_floor, target_sizes=[(h, w)])[0]
        raw_boxes = post["boxes"].cpu().numpy().tolist()
        raw_scores = post["scores"].cpu().numpy().tolist()
        raw_labels = post["labels"].cpu().numpy().tolist()
        # remap predicted labels into canonical id space; drop excluded classes
        pred_boxes, pred_scores, pred_labels = [], [], []
        for b, s, l in zip(raw_boxes, raw_scores, raw_labels):
            cid = raw2cid.get(int(l))
            if cid is None:
                continue
            pred_boxes.append(b); pred_scores.append(s); pred_labels.append(cid)
        n_pred += len(pred_labels)

        obj = row["objects"]
        gboxes, glabels = [], []
        for bbox, cat in zip(obj["bbox"], obj["category"]):
            name = canon(gt_names[int(cat)])
            n_gt_boxes += 1
            if name not in label2id:
                n_gt_dropped += 1
                continue
            cid = label2id[name]
            x, y, bw, bh = (float(v) for v in bbox)
            gboxes.append([x, y, x + bw, y + bh])
            glabels.append(cid)
            gt_per_class[cid] = gt_per_class.get(cid, 0) + 1

        recs, img_fn = _match_image(
            pred_boxes, pred_scores, pred_labels, gboxes, glabels, iou_thr)
        records.extend(recs)
        for cid, n in img_fn.items():
            fn_per_class[cid] = fn_per_class.get(cid, 0) + n
        if i % 50 == 0:
            print(f"  [{model_id}] {i}/{len(ds)}", flush=True)

    n_gt = sum(gt_per_class.values())
    print(f"[{model_id}] preds={n_pred} gt={n_gt_boxes} "
          f"({n_gt_dropped} dropped: class not in model)", flush=True)

    # ---- sweep score threshold (descending) for PR curve + AP ----
    records.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in records])
    is_tp = np.array([1 if r[1] else 0 for r in records])
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1 - is_tp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / max(n_gt, 1)
    # AP = area under PR (recall as x). Prepend (r=0,p=1) for a clean start.
    # np.trapz was removed in NumPy 2.0 → np.trapezoid (fall back for <2.0).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    ap = float(_trapz(
        np.concatenate([[1.0], precision]),
        np.concatenate([[0.0], recall]))) if len(recall) else 0.0

    # ---- ROC over predictions (positive = TP detection) ----
    from sklearn.metrics import roc_auc_score, roc_curve
    if is_tp.sum() and (1 - is_tp).sum():
        fpr, tpr, _ = roc_curve(is_tp, scores)
        roc_auc = float(roc_auc_score(is_tp, scores))
    else:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), float("nan")

    # ---- operating point at op_threshold ----
    keep = scores >= op_threshold
    tp = int(is_tp[keep].sum())
    fp = int((1 - is_tp[keep]).sum())
    fn = n_gt - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    # ---- per-class precision/recall at op_threshold ----
    per_class = {}
    by_label = {}
    for s, t, lab in records:
        by_label.setdefault(lab, []).append((s, t))
    all_labels = set(by_label) | set(gt_per_class) | set(fn_per_class)
    for lab in all_labels:
        recs = [r for r in by_label.get(lab, []) if r[0] >= op_threshold]
        ctp = sum(1 for _, t in recs if t)
        cfp = len(recs) - ctp
        cgt = gt_per_class.get(lab, 0)
        cprec = ctp / max(ctp + cfp, 1)
        crec = ctp / max(cgt, 1)
        per_class[id2label.get(lab, str(lab))] = {
            "precision": round(cprec, 4), "recall": round(crec, 4),
            "tp": ctp, "fp": cfp, "n_gt": cgt,
        }

    return {
        "model": model_id,
        "n_pred": n_pred, "n_gt": n_gt, "n_gt_dropped": n_gt_dropped,
        "op_threshold": op_threshold, "iou": iou_thr,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "ap": round(ap, 4), "roc_auc": round(roc_auc, 4),
        "per_class": per_class,
        "_curves": {
            "pr": (recall.tolist(), precision.tolist()),
            "roc": (fpr.tolist(), tpr.tolist()),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True,
                   help="Comma-separated detector repo ids.")
    p.add_argument("--dataset", default="Francesco/road-signs-6ih4y")
    p.add_argument("--dataset-config", default="default")
    p.add_argument("--split", default="test")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for a prediction to count as a TP.")
    p.add_argument("--op-threshold", type=float, default=0.5,
                   help="Confidence threshold for precision/recall/F1.")
    p.add_argument("--score-floor", type=float, default=0.0,
                   help="Post-process score floor; 0.0 keeps the full curve.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--class-aliases", default="ped_crossing:ped_zebra_cross",
                   help="Comma list of name:canonical pairs, applied to BOTH "
                        "model + GT class names so divergently-named-but-same "
                        "classes match. Default fixes road-signs' "
                        "ped_crossing/ped_zebra_cross split; pass '' to disable.")
    p.add_argument("--exclude-classes", default="",
                   help="Comma list of (canonical) class names to drop from "
                        "both predictions and GT entirely.")
    p.add_argument("--output-repo", default=None,
                   help="HF dataset repo to upload plots + metrics.json to.")
    args = p.parse_args()

    aliases = {}
    for pair in (args.class_aliases or "").split(","):
        pair = pair.strip()
        if pair and ":" in pair:
            src, dst = pair.split(":", 1)
            aliases[src.strip()] = dst.strip()
    exclude = {c.strip() for c in (args.exclude_classes or "").split(",") if c.strip()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    ds = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    obj_feat = ds.features["objects"]
    cat_feat = (obj_feat["category"] if isinstance(obj_feat, dict)
                else obj_feat.feature["category"])
    gt_names = cat_feat.feature.names if hasattr(cat_feat, "feature") else cat_feat.names
    print(f"Eval split '{args.split}' rows={len(ds)}; GT {len(gt_names)} names",
          flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = [
        evaluate_model(m, ds, gt_names, args.iou, args.op_threshold,
                       args.score_floor, aliases=aliases, exclude=exclude)
        for m in models
    ]

    # ---- headline table ----
    print("\n=== Precision / Recall / F1  (vs HUMAN GT) ===", flush=True)
    print(f"  IoU={args.iou}  op_threshold={args.op_threshold}", flush=True)
    print(f"  {'model':52s} {'P':>7} {'R':>7} {'F1':>7} {'AP':>7} {'ROC-AUC':>8}",
          flush=True)
    for r in results:
        print(f"  {r['model']:52s} {r['precision']:7.4f} {r['recall']:7.4f} "
              f"{r['f1']:7.4f} {r['ap']:7.4f} {r['roc_auc']:8.4f}", flush=True)

    for r in results:
        print(f"\n=== {r['model']} — per-class (P / R @ op) ===", flush=True)
        for name, m in sorted(r["per_class"].items(),
                              key=lambda kv: -kv[1]["recall"]):
            print(f"  {name:18s} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"(tp={m['tp']} fp={m['fp']} gt={m['n_gt']})", flush=True)

    # ---- plots ----
    def _save_fig(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    fig_pr, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        rec, prec = r["_curves"]["pr"]
        ax.plot(rec, prec, label=f"{r['model'].split('/')[-1]} (AP={r['ap']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR curve vs human GT (IoU={args.iou})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    pr_png = _save_fig(fig_pr)

    fig_roc, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr = r["_curves"]["roc"]
        ax.plot(fpr, tpr, label=f"{r['model'].split('/')[-1]} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR  (false detections kept)")
    ax.set_ylabel("TPR  (true detections kept)")
    ax.set_title("ROC: confidence separates true vs false detections")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    roc_png = _save_fig(fig_roc)

    metrics = {
        "dataset": args.dataset, "split": args.split,
        "iou": args.iou, "op_threshold": args.op_threshold,
        "class_aliases": aliases, "excluded_classes": sorted(exclude),
        "models": [{k: v for k, v in r.items() if k != "_curves"}
                   for r in results],
    }

    if args.output_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.output_repo, repo_type="dataset", exist_ok=True)
        for name, data in [("pr_curve.png", pr_png), ("roc_curve.png", roc_png)]:
            api.upload_file(path_or_fileobj=data, path_in_repo=name,
                            repo_id=args.output_repo, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=json.dumps(metrics, indent=2).encode(),
            path_in_repo="metrics.json",
            repo_id=args.output_repo, repo_type="dataset")
        readme = _build_readme(metrics)
        api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                        repo_id=args.output_repo, repo_type="dataset")
        print(f"\nUploaded plots + metrics → "
              f"https://huggingface.co/datasets/{args.output_repo}", flush=True)
    else:
        for name, data in [("pr_curve.png", pr_png), ("roc_curve.png", roc_png)]:
            with open(name, "wb") as f:
                f.write(data)
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nWrote pr_curve.png, roc_curve.png, metrics.json locally",
              flush=True)

    print("\nEVAL PR/ROC DONE", flush=True)


def _build_readme(metrics: dict) -> str:
    lines = [
        "---", "tags: [evaluation, object-detection]", "---",
        f"# PR / ROC eval vs human GT — `{metrics['dataset']}` (`{metrics['split']}`)",
        "",
        f"IoU={metrics['iou']}, operating threshold={metrics['op_threshold']}. "
        "Predictions matched greedily to human `objects` GT by class + IoU.",
        "",
        "![PR curve](pr_curve.png)",
        "![ROC curve](roc_curve.png)",
        "",
        "| model | Precision | Recall | F1 | AP | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for m in metrics["models"]:
        lines.append(
            f"| `{m['model']}` | {m['precision']} | {m['recall']} | "
            f"{m['f1']} | {m['ap']} | {m['roc_auc']} |")
    lines += [
        "",
        "**ROC note.** Detection has no natural true-negative pool, so the ROC "
        "treats each *prediction* as one sample (TP=1 / FP=0) with the model "
        "confidence as the score — it measures how well confidence separates "
        "real detections from false ones, not 1-specificity over all boxes. The "
        "PR curve / AP are the box-quality numbers.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
