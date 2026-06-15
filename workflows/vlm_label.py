"""Auto-label images for object detection using a Vision Language Model.

Accepts **local image directories** or **Hugging Face datasets** as input.

Local mode::

    python -m workflows.vlm_label \\
        --source data/images --classes "cat,dog" --output annotations.json

Hub mode::

    python -m workflows.vlm_label \\
        --source merve/my-images --classes "cat,dog" \\
        --output merve/my-images-labeled --push-to-hub \\
        --backend openai --api-key hf_...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from tools.utils import load_image
from tools.vlm_detect import vlm_detect

DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


# ------------------------------------------------------------------
# Local mode  (image_dir → COCO JSON)
# ------------------------------------------------------------------

def _label_local(
    image_dir: Path,
    classes: list[str],
    output_path: Path,
    model_id: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
) -> dict:
    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXT
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    categories = [{"id": i, "name": n} for i, n in enumerate(classes)]
    name2id = {c["name"]: c["id"] for c in categories}

    coco: dict[str, list] = {
        "images": [], "annotations": [], "categories": categories,
    }
    ann_id = 0

    for img_id, path in enumerate(tqdm(image_files, desc="Labeling")):
        image = load_image(path)
        w, h = image.size

        detections = vlm_detect(
            image, classes=classes, model_id=model_id,
            backend=backend, base_url=base_url, api_key=api_key,
        )

        coco["images"].append(
            {"id": img_id, "file_name": path.name, "width": w, "height": h},
        )
        for det in detections:
            label = det["label"].lower()
            if label not in name2id:
                continue
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": name2id[label],
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
            })
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    print(f"Labeled {len(coco['images'])} images → {ann_id} annotations → {output_path}")
    return coco


# ------------------------------------------------------------------
# Hub mode  (HF dataset → HF dataset with detections column)
# ------------------------------------------------------------------

def _label_hub(
    dataset_id: str,
    classes: list[str],
    output_id: str,
    model_id: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
    image_column: str,
    split: str,
    max_samples: int | None,
    push_to_hub: bool,
    hf_token: str | None,
    dataset_config: str | None = None,
) -> Any:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=dataset_config, split=split)

    if image_column not in ds.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {ds.column_names}"
        )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    from tools.bbox_viz import draw_detections

    all_detections: list[list[dict]] = []
    all_overlays: list[Image.Image] = []
    class_set = set(classes)

    for row in tqdm(ds, desc="Labeling", total=len(ds)):
        img = row[image_column]
        if not isinstance(img, Image.Image):
            img = load_image(img)

        try:
            dets = vlm_detect(
                img, classes=classes, model_id=model_id,
                backend=backend, base_url=base_url, api_key=api_key,
            )
        except Exception as e:
            print(f"\n  [warn] vlm_detect failed, skipping: {e}")
            dets = []
        dets = [d for d in dets if d.get("label", "").lower() in class_set]
        all_detections.append(dets)
        # Persist a numbered box-overlay render so the judge stage can score by
        # looking at the boxes drawn on the image instead of raw coordinates.
        all_overlays.append(draw_detections(img, dets, show_index=True))

    from datasets import Image as HFImage

    ds = ds.add_column("detections", all_detections)
    ds = ds.add_column("detections_overlay", all_overlays)
    ds = ds.cast_column("detections_overlay", HFImage())

    if push_to_hub:
        from tools.hub_viz import push_dataset_with_viz
        push_dataset_with_viz(
            ds, output_id, token=hf_token, image_column=image_column,
            detections_column="detections",
        )
        print(f"Labeled {len(ds)} rows → https://huggingface.co/datasets/{output_id}")
    else:
        ds.save_to_disk(output_id)
        print(f"Labeled {len(ds)} rows → {output_id}")

    return ds


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def label_dataset(
    source: str | Path,
    classes: list[str],
    output: str | Path,
    model_id: str = DEFAULT_VLM,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    image_column: str = "image",
    split: str = "train",
    max_samples: int | None = None,
    push_to_hub: bool = False,
    hf_token: str | None = None,
    dataset_config: str | None = None,
) -> dict | Any:
    """Auto-label images and write detection annotations.

    Parameters
    ----------
    source : str or Path
        **Local directory** of images, or a **Hugging Face dataset ID**
        (e.g. ``"merve/my-images"``).  Detected automatically.
    classes : list[str]
        Object categories the VLM should look for.
    output : str or Path
        Local COCO JSON path (local mode) or HF dataset ID (Hub mode).
    model_id : str
        Model identifier.
    backend / base_url / api_key
        Inference backend (see :mod:`tools.vlm_client`).
    image_column : str
        Image column name in the HF dataset (Hub mode only).
    split : str
        Dataset split (Hub mode only).
    max_samples : int, optional
        Cap the number of images to process.
    push_to_hub : bool
        If ``True``, push the labeled dataset to the Hub.
    hf_token : str, optional
        Hugging Face token for pushing.
    dataset_config : str, optional
        Dataset configuration name (Hub mode only).  Required for
        multi-config datasets like ``lmms-lab/DocVQA``.

    Returns
    -------
    dict or datasets.Dataset
        COCO dict (local mode) or HF Dataset with ``detections``
        column (Hub mode).
    """
    classes_lower = [c.lower() for c in classes]

    if _is_local_dir(source):
        return _label_local(
            Path(source), classes_lower, Path(output),
            model_id, backend, base_url, api_key,
        )
    return _label_hub(
        str(source), classes_lower, str(output),
        model_id, backend, base_url, api_key,
        image_column, split, max_samples, push_to_hub, hf_token,
        dataset_config=dataset_config,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Auto-label images with a VLM")
    parser.add_argument("--source", required=True,
                        help="Image directory or HF dataset ID")
    parser.add_argument("--classes", required=True,
                        help="Comma-separated object categories")
    parser.add_argument("--output", required=True,
                        help="COCO JSON path or HF dataset ID")
    parser.add_argument("--model", default=DEFAULT_VLM)
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset config name (multi-config datasets)")
    args = parser.parse_args()

    label_dataset(
        source=args.source,
        classes=[c.strip() for c in args.classes.split(",")],
        output=args.output,
        model_id=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        image_column=args.image_column,
        split=args.split,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    _cli()
