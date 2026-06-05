"""Fine-tune RF-DETR on a detection dataset.

Accepts **local COCO-format directories** or **Hugging Face datasets**
(with a ``detections`` column from :func:`workflows.vlm_label.label_dataset`
or the standard ``objects`` column used by HF detection datasets).

Local mode::

    python -m workflows.train_rfdetr --source data/ --epochs 50

Hub mode::

    python -m workflows.train_rfdetr \\
        --source merve/my-dataset-filtered \\
        --epochs 50 --output-dir checkpoints/rfdetr
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_MODEL = "Roboflow/rf-detr-base"


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


# ------------------------------------------------------------------
# Local COCO dataset (unchanged from before)
# ------------------------------------------------------------------

class CocoDetectionDataset(Dataset):
    """Minimal COCO-format dataset for DETR-family training."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        processor: Any,
    ) -> None:
        self.image_dir = Path(image_dir)
        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.processor = processor

        self.img2anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.img2anns[ann["image_id"]].append(ann)

        old_ids = sorted({c["id"] for c in coco["categories"]})
        self.cat_remap = {old: new for new, old in enumerate(old_ids)}
        self.categories = coco["categories"]

    @property
    def num_labels(self) -> int:
        return len(self.categories)

    @property
    def id2label(self) -> dict[int, str]:
        return {self.cat_remap[c["id"]]: c["name"] for c in self.categories}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.images[idx]
        image = Image.open(
            self.image_dir / img_info["file_name"],
        ).convert("RGB")

        anns = self.img2anns[img_info["id"]]
        target = {
            "image_id": img_info["id"],
            "annotations": [
                {
                    "bbox": a["bbox"],
                    "category_id": self.cat_remap[a["category_id"]],
                    "area": a.get("area", a["bbox"][2] * a["bbox"][3]),
                    "iscrowd": a.get("iscrowd", 0),
                }
                for a in anns
            ],
        }

        encoding = self.processor(
            images=image, annotations=[target], return_tensors="pt",
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


# ------------------------------------------------------------------
# Hub dataset  (HF dataset with detections / objects column)
# ------------------------------------------------------------------

def _discover_categories(hf_dataset: Any, detections_col: str) -> list[dict]:
    """Scan the dataset to collect all unique labels → category list."""
    labels: set[str] = set()
    for row in hf_dataset:
        for det in row.get(detections_col, []) or []:
            lab = det.get("label", "")
            if lab:
                labels.add(lab.lower())
    return [{"id": i, "name": n} for i, n in enumerate(sorted(labels))]


def _objects_to_detections(hf_dataset: Any) -> Any:
    """Convert HF standard ``objects`` column to flat ``detections`` list."""
    def _convert(row: dict) -> dict:
        objects = row["objects"]
        bboxes = objects.get("bbox", [])
        cats = objects.get("category", [])
        dets = []
        for bbox, cat in zip(bboxes, cats):
            label = cat if isinstance(cat, str) else str(cat)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            dets.append({"bbox": [x1, y1, x2, y2], "label": label, "sub_label": ""})
        return {"detections": dets}

    return hf_dataset.map(_convert)


class HubDetectionDataset(Dataset):
    """Wrap an HF ``datasets.Dataset`` with a ``detections`` column."""

    def __init__(
        self,
        hf_dataset: Any,
        processor: Any,
        image_column: str,
        detections_column: str,
        categories: list[dict],
    ) -> None:
        self.dataset = hf_dataset
        self.processor = processor
        self.image_column = image_column
        self.detections_column = detections_column
        self.categories = categories
        self._label2id = {c["name"]: c["id"] for c in categories}

    @property
    def num_labels(self) -> int:
        return len(self.categories)

    @property
    def id2label(self) -> dict[int, str]:
        return {c["id"]: c["name"] for c in self.categories}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]

        img = row[self.image_column]
        if not isinstance(img, Image.Image):
            import io as _io
            img = Image.open(_io.BytesIO(img["bytes"])).convert("RGB")
        else:
            img = img.convert("RGB")

        dets = row.get(self.detections_column, []) or []

        annotations = []
        for det in dets:
            label = det.get("label", "").lower()
            if label not in self._label2id:
                continue
            bbox = det["bbox"]
            x1, y1, x2, y2 = (float(v) for v in bbox)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "bbox": [x1, y1, w, h],
                "category_id": self._label2id[label],
                "area": w * h,
                "iscrowd": 0,
            })

        target = {"image_id": idx, "annotations": annotations}
        encoding = self.processor(
            images=img, annotations=[target], return_tensors="pt",
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def _build_optimizer(
    model: Any, lr: float, backbone_factor: float,
) -> torch.optim.Optimizer:
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "encoder" in name:
            backbone.append(param)
        else:
            head.append(param)
    return torch.optim.AdamW([
        {"params": backbone, "lr": lr * backbone_factor},
        {"params": head, "lr": lr},
    ], weight_decay=0.01)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train(
    source: str | Path,
    output_dir: str | Path = "checkpoints/rfdetr-finetuned",
    model_id: str = DEFAULT_MODEL,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-5,
    backbone_lr_factor: float = 0.1,
    gradient_accumulation_steps: int = 1,
    *,
    image_column: str = "image",
    detections_column: str = "detections",
    train_split: str = "train",
    val_split: str | None = "test",
) -> Path:
    """Fine-tune RF-DETR and save the resulting model.

    Parameters
    ----------
    source : str or Path
        **Local directory** with ``train/images/`` + ``train/labels.json``
        (COCO format), or a **Hugging Face dataset ID** with an image
        column and a ``detections`` (from :func:`~workflows.vlm_label.label_dataset`)
        or ``objects`` column (standard HF detection format).
    output_dir : path
        Where to save the fine-tuned model.
    model_id : str
        Pretrained RF-DETR checkpoint on the Hub.
    epochs / batch_size / lr / backbone_lr_factor / gradient_accumulation_steps
        Training hyperparameters.
    image_column : str
        Image column name (Hub mode only).
    detections_column : str
        Detections column name (Hub mode only).
    train_split / val_split : str
        Dataset splits (Hub mode only).  Set *val_split* to ``None``
        to skip validation.

    Returns
    -------
    Path
        Directory containing the saved model.
    """
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)
    processor = AutoImageProcessor.from_pretrained(model_id)

    # ----- build train / val datasets -----
    if _is_local_dir(source):
        data_dir = Path(source)
        train_ds = CocoDetectionDataset(
            data_dir / "train" / "images",
            data_dir / "train" / "labels.json",
            processor,
        )
        val_ds = None
        val_path = data_dir / "val"
        if val_path.exists():
            val_ds = CocoDetectionDataset(
                val_path / "images", val_path / "labels.json", processor,
            )
    else:
        from datasets import load_dataset

        hf_train = load_dataset(str(source), split=train_split)

        if detections_column not in hf_train.column_names and "objects" in hf_train.column_names:
            hf_train = _objects_to_detections(hf_train)
            detections_column = "detections"

        categories = _discover_categories(hf_train, detections_column)
        if not categories:
            raise ValueError(
                f"No labels found in column '{detections_column}'. "
                "Check that the dataset has detections."
            )

        train_ds = HubDetectionDataset(
            hf_train, processor, image_column, detections_column, categories,
        )

        val_ds = None
        if val_split:
            try:
                hf_val = load_dataset(str(source), split=val_split)
                if detections_column not in hf_val.column_names and "objects" in hf_val.column_names:
                    hf_val = _objects_to_detections(hf_val)
                val_ds = HubDetectionDataset(
                    hf_val, processor, image_column, detections_column,
                    categories,
                )
            except (ValueError, KeyError):
                pass

    id2label = train_ds.id2label
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForObjectDetection.from_pretrained(
        model_id,
        num_labels=train_ds.num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        save_total_limit=3,
        load_best_model_at_end=bool(val_ds),
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_collate,
        optimizers=(_build_optimizer(model, lr, backbone_lr_factor), None),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"Model saved \u2192 {output_dir / 'final'}")
    return output_dir / "final"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR")
    parser.add_argument("--source", required=True,
                        help="Local data dir or HF dataset ID")
    parser.add_argument("--output-dir", default="checkpoints/rfdetr-finetuned")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--detections-column", default="detections")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    args = parser.parse_args()

    train(
        source=args.source,
        output_dir=args.output_dir,
        model_id=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr_factor=args.backbone_lr_factor,
        gradient_accumulation_steps=args.grad_accum,
        image_column=args.image_column,
        detections_column=args.detections_column,
        train_split=args.train_split,
        val_split=args.val_split or None,
    )


if __name__ == "__main__":
    _cli()
