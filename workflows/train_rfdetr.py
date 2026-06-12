"""Fine-tune an object-detection model (RF-DETR by default).

This is a generalized version of the Hugging Face object-detection tutorial
(https://huggingface.co/docs/transformers/tasks/object_detection): lazy
``with_transform`` preprocessing, optional Albumentations augmentation, and
COCO-style **mAP / mAR** evaluation via ``torchmetrics`` — wired up so it
works with any of three input formats:

* **HF dataset, ``objects`` column** — the standard HF detection layout
  (``objects = {bbox: [x,y,w,h] (COCO), category: int|str, ...}``).
* **HF dataset, ``detections`` column** — produced by
  :func:`workflows.vlm_label.label_dataset` (``[{bbox: [x1,y1,x2,y2],
  label: str}]``, Pascal-VOC boxes).
* **Local COCO directory** — ``train/images/`` + ``train/labels.json``
  (and optional ``val/``).

Any ``AutoModelForObjectDetection`` checkpoint works; the default is
``Roboflow/rf-detr-base``.

Examples
--------
Hub::

    python -m workflows.train_rfdetr \\
        --source merve/docvqa-media-judged --train-split test \\
        --epochs 10 --output-dir checkpoints/rfdetr

Local COCO::

    python -m workflows.train_rfdetr --source data/ --epochs 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_MODEL = "Roboflow/rf-detr-base"


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


# ==================================================================
# 1. Load any supported input into a normalized HF dataset
# ------------------------------------------------------------------
# Normalized schema (one row per image):
#   image     : PIL.Image  (lazy-decoded datasets ``Image`` feature)
#   image_id  : int
#   objects   : {bbox: list[[x, y, w, h]] (COCO),
#                category: list[int],
#                area: list[float]}
# ==================================================================


def _normalize_objects_row(example: dict, idx: int, label2id: dict[str, int]) -> dict:
    """Standard ``objects`` column → normalized, with clipped/clean boxes."""
    objects = example["objects"]
    bboxes = objects.get("bbox", [])
    cats = objects.get("category", [])
    img = example["image"]
    img_w, img_h = img.size if hasattr(img, "size") else (
        example.get("width"), example.get("height"))

    out_boxes, out_cats, out_areas = [], [], []
    for bbox, cat in zip(bboxes, cats):
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            continue
        x = max(0.0, min(x, img_w))
        y = max(0.0, min(y, img_h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w <= 0 or h <= 0:
            continue
        cat_id = cat if isinstance(cat, int) else label2id[str(cat).lower()]
        out_boxes.append([x, y, w, h])
        out_cats.append(cat_id)
        out_areas.append(w * h)
    return {
        "image": img,
        "image_id": idx,
        "objects": {"bbox": out_boxes, "category": out_cats, "area": out_areas},
    }


def _normalize_detections_row(example: dict, idx: int, label2id: dict[str, int],
                              detections_column: str) -> dict:
    """Pipeline ``detections`` column (xyxy) → normalized COCO xywh."""
    img = example["image"]
    img_w, img_h = img.size
    dets = example.get(detections_column, []) or []

    out_boxes, out_cats, out_areas = [], [], []
    for det in dets:
        label = str(det.get("label", "")).lower()
        if label not in label2id:
            continue
        x1, y1, x2, y2 = (float(v) for v in det["bbox"])
        x, y = max(0.0, x1), max(0.0, y1)
        w, h = min(x2, img_w) - x, min(y2, img_h) - y
        if w <= 0 or h <= 0:
            continue
        out_boxes.append([x, y, w, h])
        out_cats.append(label2id[label])
        out_areas.append(w * h)
    return {
        "image": img,
        "image_id": idx,
        "objects": {"bbox": out_boxes, "category": out_cats, "area": out_areas},
    }


def _categories_from_objects(ds: Any) -> tuple[dict[int, str], dict[str, int]]:
    """Derive id2label/label2id from a standard ``objects`` dataset.

    Uses the ``ClassLabel`` feature names when available, otherwise scans the
    string categories present in the data.
    """
    feat = ds.features["objects"]
    cat_feat = feat.feature["category"] if hasattr(feat, "feature") else None
    names = getattr(cat_feat, "names", None)
    if names:
        id2label = {i: n for i, n in enumerate(names)}
        return id2label, {n: i for i, n in id2label.items()}

    labels: set[str] = set()
    for row in ds:
        for cat in row["objects"].get("category", []):
            labels.add(str(cat).lower())
    id2label = {i: n for i, n in enumerate(sorted(labels))}
    return id2label, {n: i for i, n in id2label.items()}


def _categories_from_detections(ds: Any, col: str) -> tuple[dict[int, str], dict[str, int]]:
    labels: set[str] = set()
    for row in ds:
        for det in row.get(col, []) or []:
            lab = str(det.get("label", "")).lower()
            if lab:
                labels.add(lab)
    if not labels:
        raise ValueError(
            f"No labels found in column '{col}'. Check that the dataset has detections."
        )
    id2label = {i: n for i, n in enumerate(sorted(labels))}
    return id2label, {n: i for i, n in id2label.items()}


def _load_local_coco(data_dir: Path, split_dir: str) -> tuple[Any, dict, dict]:
    """Build a normalized HF dataset from a COCO ``split_dir`` folder."""
    from datasets import Dataset, Features, Image, Sequence, Value

    img_dir = data_dir / split_dir / "images"
    with open(data_dir / split_dir / "labels.json") as f:
        coco = json.load(f)

    old_ids = sorted({c["id"] for c in coco["categories"]})
    remap = {old: new for new, old in enumerate(old_ids)}
    id2label = {remap[c["id"]]: c["name"] for c in coco["categories"]}
    label2id = {v: k for k, v in id2label.items()}

    from collections import defaultdict
    img2anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        img2anns[ann["image_id"]].append(ann)

    rows = {"image": [], "image_id": [], "objects": []}
    for img_info in coco["images"]:
        anns = img2anns.get(img_info["id"], [])
        boxes, cats, areas = [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])
            cats.append(remap[a["category_id"]])
            areas.append(float(a.get("area", w * h)))
        rows["image"].append(str(img_dir / img_info["file_name"]))
        rows["image_id"].append(img_info["id"])
        rows["objects"].append({"bbox": boxes, "category": cats, "area": areas})

    features = Features({
        "image": Image(),
        "image_id": Value("int64"),
        "objects": {
            "bbox": Sequence(Sequence(Value("float32"))),
            "category": Sequence(Value("int64")),
            "area": Sequence(Value("float32")),
        },
    })
    ds = Dataset.from_dict(rows, features=features)
    return ds, id2label, label2id


def _load_normalized(
    source: str | Path,
    train_split: str,
    val_split: str | None,
    image_column: str,
    detections_column: str,
    val_size: float,
    seed: int,
) -> tuple[Any, Any, dict[int, str], dict[str, int]]:
    if _is_local_dir(source):
        data_dir = Path(source)
        train_ds, id2label, label2id = _load_local_coco(data_dir, "train")
        val_ds = None
        if (data_dir / "val").exists():
            val_ds, _, _ = _load_local_coco(data_dir, "val")
        return train_ds, val_ds, id2label, label2id

    from datasets import load_dataset

    raw = load_dataset(str(source), split=train_split)
    if image_column != "image":
        raw = raw.rename_column(image_column, "image")

    cols = raw.column_names
    if "objects" in cols:
        id2label, label2id = _categories_from_objects(raw)
        norm = raw.map(
            partial(_normalize_objects_row, label2id=label2id),
            with_indices=True, remove_columns=cols,
        )
    elif detections_column in cols:
        id2label, label2id = _categories_from_detections(raw, detections_column)
        norm = raw.map(
            partial(_normalize_detections_row, label2id=label2id,
                    detections_column=detections_column),
            with_indices=True, remove_columns=cols,
        )
    else:
        raise ValueError(
            f"Dataset has neither an 'objects' nor a '{detections_column}' column. "
            f"Found: {cols}"
        )

    norm = norm.filter(lambda x: len(x["objects"]["bbox"]) > 0)

    if val_split:
        try:
            raw_val = load_dataset(str(source), split=val_split)
            if image_column != "image":
                raw_val = raw_val.rename_column(image_column, "image")
            vcols = raw_val.column_names
            if "objects" in vcols:
                val_ds = raw_val.map(
                    partial(_normalize_objects_row, label2id=label2id),
                    with_indices=True, remove_columns=vcols,
                ).filter(lambda x: len(x["objects"]["bbox"]) > 0)
            else:
                val_ds = raw_val.map(
                    partial(_normalize_detections_row, label2id=label2id,
                            detections_column=detections_column),
                    with_indices=True, remove_columns=vcols,
                ).filter(lambda x: len(x["objects"]["bbox"]) > 0)
            return norm, val_ds, id2label, label2id
        except (ValueError, KeyError):
            pass
        return norm, None, id2label, label2id

    if val_size and val_size > 0:
        split = norm.train_test_split(test_size=val_size, seed=seed)
        return split["train"], split["test"], id2label, label2id

    return norm, None, id2label, label2id


# ==================================================================
# 2. Preprocessing transforms (tutorial)
# ==================================================================


def _format_anns_as_coco(image_id: int, categories, areas, bboxes) -> dict:
    annotations = [
        {"image_id": image_id, "category_id": cat, "iscrowd": 0,
         "area": area, "bbox": list(bbox)}
        for cat, area, bbox in zip(categories, areas, bboxes)
    ]
    return {"image_id": image_id, "annotations": annotations}


def _transform_batch(examples, image_processor) -> dict:
    images, annotations = [], []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        images.append(np.array(image.convert("RGB")))
        annotations.append(_format_anns_as_coco(
            image_id, objects["category"], objects["area"], objects["bbox"]))
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    result.pop("pixel_mask", None)
    return result


def _augment_and_transform_batch(examples, image_processor, transform) -> dict:
    images, annotations = [], []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(out["image"])
        areas = [w * h for (_, _, w, h) in out["bboxes"]]
        annotations.append(_format_anns_as_coco(
            image_id, out["category"], areas, out["bboxes"]))
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    result.pop("pixel_mask", None)
    return result


def _build_augment():
    """Albumentations train-time augmentation, or ``None`` if unavailable."""
    try:
        import albumentations as A
    except ImportError:
        print("[warn] albumentations not installed → training without augmentation. "
              "`pip install albumentations` to enable it.")
        return None
    return A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], clip=True, min_area=25),
    )


def _collate(batch: list[dict]) -> dict:
    data = {"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": [x["labels"] for x in batch]}
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


# ==================================================================
# 3. mAP / mAR metrics (tutorial)
# ==================================================================


def _convert_bbox_yolo_to_pascal(boxes, image_size):
    from transformers.image_transforms import center_to_corners_format
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes


@dataclass
class _ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def _get_orig_size(image_target):
    orig = np.atleast_1d(np.asarray(image_target["orig_size"])).flatten()
    if len(orig) >= 2:
        return int(orig[0]), int(orig[1])
    return int(orig[0]), int(orig[0])


@torch.no_grad()
def _compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    image_sizes, post_targets, post_preds = [], [], []

    for batch in targets:
        batch_sizes = []
        for image_target in batch:
            h, w = _get_orig_size(image_target)
            batch_sizes.append([h, w])
            boxes = _convert_bbox_yolo_to_pascal(torch.tensor(image_target["boxes"]), (h, w))
            labels = torch.tensor(image_target["class_labels"])
            post_targets.append({"boxes": boxes, "labels": labels})
        image_sizes.append(torch.tensor(batch_sizes))

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = _ModelOutput(
            logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_preds.extend(image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes))

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_preds, post_targets)
    metrics = metric.compute()

    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{name}"] = class_map
        metrics[f"mar_100_{name}"] = class_mar
    return {k: round(v.item(), 4) for k, v in metrics.items()}


# ==================================================================
# 4. Public API
# ==================================================================


def train(
    source: str | Path,
    output_dir: str | Path = "checkpoints/rfdetr-finetuned",
    model_id: str = DEFAULT_MODEL,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    *,
    augment: bool = True,
    image_column: str = "image",
    detections_column: str = "detections",
    train_split: str = "train",
    val_split: str | None = "test",
    val_size: float = 0.0,
    seed: int = 1337,
    num_workers: int = 4,
    eval_threshold: float = 0.0,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    report_to: str = "none",
    run_name: str | None = None,
) -> Path:
    """Fine-tune an object detector and save it.

    Parameters
    ----------
    source : str or Path
        Local COCO directory (``train/`` [+ ``val/``]) or a Hugging Face
        dataset ID with an ``objects`` or ``detections`` column.
    output_dir : path
        Where checkpoints / the final model are written.
    model_id : str
        Any ``AutoModelForObjectDetection`` checkpoint (default RF-DETR base).
    epochs / batch_size / lr / weight_decay / gradient_accumulation_steps
        Training hyperparameters.
    augment : bool
        Apply Albumentations augmentation to the train split (no-op if the
        package is missing).
    image_column / detections_column : str
        Column names (Hub mode).
    train_split / val_split : str
        Dataset splits (Hub mode). Set *val_split* to ``None`` to skip the
        Hub validation split.
    val_size : float
        If no validation set is available, hold out this fraction of train
        for evaluation (``0`` disables evaluation).
    eval_threshold : float
        Confidence threshold used when post-processing predictions for mAP.
    push_to_hub : bool
        Push checkpoints + final model to the Hub.
    hub_model_id : str, optional
        Target repo for ``push_to_hub``.
    report_to / run_name : str
        Experiment tracking (e.g. ``"trackio"``).

    Returns
    -------
    Path
        Directory containing the saved final model.
    """
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)

    train_ds, val_ds, id2label, label2id = _load_normalized(
        source, train_split, val_split, image_column, detections_column,
        val_size, seed,
    )
    print(f"Categories ({len(id2label)}): {list(id2label.values())}")
    print(f"Train: {len(train_ds)}" + (f", Val: {len(val_ds)}" if val_ds else " (no eval)"))

    image_processor = AutoImageProcessor.from_pretrained(model_id)

    transform_fn = partial(_transform_batch, image_processor=image_processor)
    aug = _build_augment() if augment else None
    if aug is not None:
        train_ds = train_ds.with_transform(
            partial(_augment_and_transform_batch,
                    image_processor=image_processor, transform=aug))
    else:
        train_ds = train_ds.with_transform(transform_fn)
    if val_ds is not None:
        val_ds = val_ds.with_transform(transform_fn)

    model = AutoModelForObjectDetection.from_pretrained(
        model_id,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    has_eval = val_ds is not None
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=num_workers,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        max_grad_norm=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_map" if has_eval else None,
        greater_is_better=True if has_eval else None,
        load_best_model_at_end=has_eval,
        eval_do_concat_batches=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    compute_metrics = (
        partial(_compute_metrics, image_processor=image_processor,
                id2label=id2label, threshold=eval_threshold)
        if has_eval else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        data_collator=_collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    image_processor.save_pretrained(str(final_dir))
    if push_to_hub:
        trainer.push_to_hub()
    print(f"Model saved \u2192 {final_dir}")
    return final_dir


# ==================================================================
# CLI
# ==================================================================


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an object detector (RF-DETR)")
    parser.add_argument("--source", required=True,
                        help="Local COCO dir or HF dataset ID")
    parser.add_argument("--output-dir", default="checkpoints/rfdetr-finetuned")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable Albumentations augmentation")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--detections-column", default="detections")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test",
                        help="Hub validation split ('none' to disable)")
    parser.add_argument("--val-size", type=float, default=0.0,
                        help="Hold out this fraction of train for eval if no val split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-threshold", type=float, default=0.0)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    val_split = None if (args.val_split or "").lower() == "none" else args.val_split

    train(
        source=args.source,
        output_dir=args.output_dir,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        augment=not args.no_augment,
        image_column=args.image_column,
        detections_column=args.detections_column,
        train_split=args.train_split,
        val_split=val_split,
        val_size=args.val_size,
        num_workers=args.num_workers,
        eval_threshold=args.eval_threshold,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to=args.report_to,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    _cli()
