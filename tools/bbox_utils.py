"""Bounding-box format conversion, annotation validation, and dataset statistics.

Adapted from `uv-scripts/object-detection
<https://huggingface.co/datasets/uv-scripts/object-detection>`_.
Supports 6 bbox formats used across the detection ecosystem.  All functions
are CPU-only — no model loading required.

CLI usage::

    python -m tools.bbox_utils convert --from coco_xywh --to yolo \\
        --bbox "[10, 20, 100, 50]" --img-w 640 --img-h 480

Supported formats
-----------------
=============  ==================================  ================
Format         Encoding                            Coordinate space
=============  ==================================  ================
coco_xywh      [x, y, width, height]               pixels
xyxy           [xmin, ymin, xmax, ymax]             pixels
voc            [xmin, ymin, xmax, ymax]             pixels (alias)
yolo           [cx, cy, w, h]                       normalised 0-1
tfod           [xmin, ymin, xmax, ymax]             normalised 0-1
label_studio   [x, y, width, height]                percentage 0-100
=============  ==================================  ================
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any

BBOX_FORMATS = ("coco_xywh", "xyxy", "voc", "yolo", "tfod", "label_studio")


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------

def convert_bbox(
    bbox: list[float],
    from_fmt: str,
    to_fmt: str,
    img_w: float = 1.0,
    img_h: float = 1.0,
) -> list[float]:
    """Convert a single bbox between any two of the 6 supported formats.

    All conversions route through XYXY pixel-space as intermediate.
    """
    # → XYXY pixel-space
    if from_fmt == "coco_xywh":
        x, y, w, h = bbox[:4]
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif from_fmt in ("xyxy", "voc"):
        xmin, ymin, xmax, ymax = bbox[:4]
    elif from_fmt == "yolo":
        cx, cy, w, h = bbox[:4]
        xmin = (cx - w / 2) * img_w
        ymin = (cy - h / 2) * img_h
        xmax = (cx + w / 2) * img_w
        ymax = (cy + h / 2) * img_h
    elif from_fmt == "tfod":
        xmin = bbox[0] * img_w
        ymin = bbox[1] * img_h
        xmax = bbox[2] * img_w
        ymax = bbox[3] * img_h
    elif from_fmt == "label_studio":
        x_p, y_p, w_p, h_p = bbox[:4]
        xmin = x_p / 100.0 * img_w
        ymin = y_p / 100.0 * img_h
        xmax = (x_p + w_p) / 100.0 * img_w
        ymax = (y_p + h_p) / 100.0 * img_h
    else:
        raise ValueError(f"Unknown source format: {from_fmt}")

    # XYXY → target
    if to_fmt in ("xyxy", "voc"):
        return [xmin, ymin, xmax, ymax]
    if to_fmt == "coco_xywh":
        return [xmin, ymin, xmax - xmin, ymax - ymin]
    if to_fmt == "yolo":
        w, h = xmax - xmin, ymax - ymin
        return [(xmin + w / 2) / img_w, (ymin + h / 2) / img_h, w / img_w, h / img_h]
    if to_fmt == "tfod":
        return [xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h]
    if to_fmt == "label_studio":
        return [
            xmin / img_w * 100, ymin / img_h * 100,
            (xmax - xmin) / img_w * 100, (ymax - ymin) / img_h * 100,
        ]
    raise ValueError(f"Unknown target format: {to_fmt}")


def convert_annotations(
    annotations: list[dict],
    from_fmt: str,
    to_fmt: str,
    img_w: float = 1.0,
    img_h: float = 1.0,
    bbox_key: str = "bbox",
) -> list[dict]:
    """Convert the ``bbox_key`` field of every annotation in a list."""
    out = []
    for ann in annotations:
        new = dict(ann)
        if bbox_key in new and new[bbox_key] is not None and len(new[bbox_key]) >= 4:
            new[bbox_key] = convert_bbox(new[bbox_key], from_fmt, to_fmt, img_w, img_h)
        out.append(new)
    return out


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def _to_xyxy(
    bbox: list[float], fmt: str, img_w: float = 1.0, img_h: float = 1.0,
) -> tuple[float, float, float, float]:
    return tuple(convert_bbox(bbox, fmt, "xyxy", img_w, img_h))  # type: ignore[return-value]


def validate_annotations(
    annotations: list[dict],
    bbox_format: str = "coco_xywh",
    img_w: float | None = None,
    img_h: float | None = None,
    bbox_key: str = "bbox",
    category_key: str = "category_id",
    tolerance: float = 0.5,
) -> list[dict]:
    """Check a list of annotation dicts for common issues.

    Returns a list of issue dicts, each with ``level`` (error/warning),
    ``code``, ``message``, and ``annotation_idx``.

    Issue codes
    -----------
    E002  Invalid bbox (< 4 values)
    E003  Non-finite coordinates
    E004  xmin > xmax
    E005  ymin > ymax
    W002  Zero or negative area
    W003  Bbox before image origin
    W004  Bbox beyond image bounds
    W005  Empty category label
    """
    issues: list[dict] = []

    for i, ann in enumerate(annotations):
        bbox = ann.get(bbox_key)
        if bbox is None or len(bbox) < 4:
            issues.append({"level": "error", "code": "E002",
                           "message": f"Invalid bbox: {bbox}", "annotation_idx": i})
            continue

        if not all(math.isfinite(v) for v in bbox[:4]):
            issues.append({"level": "error", "code": "E003",
                           "message": f"Non-finite coords: {bbox}", "annotation_idx": i})
            continue

        w_c = img_w if img_w else 1.0
        h_c = img_h if img_h else 1.0
        xmin, ymin, xmax, ymax = _to_xyxy(bbox[:4], bbox_format, w_c, h_c)

        if xmin > xmax:
            issues.append({"level": "error", "code": "E004",
                           "message": f"xmin ({xmin}) > xmax ({xmax})", "annotation_idx": i})
        if ymin > ymax:
            issues.append({"level": "error", "code": "E005",
                           "message": f"ymin ({ymin}) > ymax ({ymax})", "annotation_idx": i})

        area = (xmax - xmin) * (ymax - ymin)
        if area <= 0:
            issues.append({"level": "warning", "code": "W002",
                           "message": f"Zero/negative area: {bbox}", "annotation_idx": i})

        if img_w is not None and img_h is not None:
            if xmin < -tolerance or ymin < -tolerance:
                issues.append({"level": "warning", "code": "W003",
                               "message": f"Before origin: ({xmin:.1f}, {ymin:.1f})",
                               "annotation_idx": i})
            if xmax > img_w + tolerance or ymax > img_h + tolerance:
                issues.append({"level": "warning", "code": "W004",
                               "message": f"Beyond bounds: ({xmax:.1f}, {ymax:.1f})",
                               "annotation_idx": i})

        cat = ann.get(category_key)
        if cat is None or (isinstance(cat, str) and cat.strip() == ""):
            issues.append({"level": "warning", "code": "W005",
                           "message": "Empty category", "annotation_idx": i})

    return issues


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (k - f) * (vals[c] - vals[f])


def compute_stats(
    coco: dict,
    bbox_format: str = "coco_xywh",
    top: int = 10,
) -> dict:
    """Compute rich statistics for a COCO-format annotation dict.

    Parameters
    ----------
    coco : dict
        Standard COCO dict with ``images``, ``annotations``, ``categories``.
    bbox_format : str
        Format of the ``bbox`` field in annotations (default ``coco_xywh``).
    top : int
        Number of entries for histograms.

    Returns
    -------
    dict
        Summary with label distribution, bbox area/aspect-ratio stats,
        annotation density, per-category areas, and co-occurrence pairs.
    """
    id2cat = {c["id"]: c["name"] for c in coco.get("categories", [])}
    id2img = {img["id"]: img for img in coco.get("images", [])}

    img2anns: dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img2anns[ann["image_id"]].append(ann)

    cat_counts: Counter = Counter()
    areas: list[float] = []
    aspect_ratios: list[float] = []
    anns_per_img: list[int] = []
    per_cat_areas: dict[str, list[float]] = defaultdict(list)
    cooccur: Counter = Counter()

    for img in coco.get("images", []):
        anns = img2anns.get(img["id"], [])
        anns_per_img.append(len(anns))
        iw = img.get("width", 1)
        ih = img.get("height", 1)
        cats_in_img: set[str] = set()

        for ann in anns:
            cname = id2cat.get(ann.get("category_id"), "unknown")
            cat_counts[cname] += 1
            cats_in_img.add(cname)

            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            xmin, ymin, xmax, ymax = _to_xyxy(bbox[:4], bbox_format, iw, ih)
            bw, bh = xmax - xmin, ymax - ymin
            a = bw * bh
            if a > 0:
                areas.append(a)
                per_cat_areas[cname].append(a)
            if bh > 0:
                aspect_ratios.append(bw / bh)

        sc = sorted(cats_in_img)
        for i in range(len(sc)):
            for j in range(i + 1, len(sc)):
                cooccur[(sc[i], sc[j])] += 1

    def _dist(v: list[float]) -> dict:
        if not v:
            return {}
        v.sort()
        return {
            "count": len(v), "min": round(v[0], 2), "max": round(v[-1], 2),
            "mean": round(sum(v) / len(v), 2),
            "median": round(_percentile(v, 50), 2),
        }

    return {
        "total_images": len(coco.get("images", [])),
        "total_annotations": len(coco.get("annotations", [])),
        "unique_categories": len(cat_counts),
        "label_distribution": dict(cat_counts.most_common(top)),
        "annotation_density": _dist([float(x) for x in sorted(anns_per_img)]),
        "bbox_area": _dist(areas),
        "bbox_aspect_ratio": _dist(aspect_ratios),
        "co_occurrence_pairs": [
            {"pair": list(p), "count": c} for p, c in cooccur.most_common(top)
        ],
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bbox utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert a single bbox")
    p_conv.add_argument("--bbox", required=True, help="JSON bbox array")
    p_conv.add_argument("--from", dest="from_fmt", required=True, choices=BBOX_FORMATS)
    p_conv.add_argument("--to", dest="to_fmt", required=True, choices=BBOX_FORMATS)
    p_conv.add_argument("--img-w", type=float, default=1.0)
    p_conv.add_argument("--img-h", type=float, default=1.0)

    p_val = sub.add_parser("validate", help="Validate COCO JSON annotations")
    p_val.add_argument("coco_json", help="Path to COCO annotation file")
    p_val.add_argument("--format", dest="bbox_fmt", default="coco_xywh", choices=BBOX_FORMATS)

    p_stats = sub.add_parser("stats", help="Compute dataset statistics")
    p_stats.add_argument("coco_json", help="Path to COCO annotation file")
    p_stats.add_argument("--format", dest="bbox_fmt", default="coco_xywh", choices=BBOX_FORMATS)
    p_stats.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "convert":
        bbox = json.loads(args.bbox)
        result = convert_bbox(bbox, args.from_fmt, args.to_fmt, args.img_w, args.img_h)
        print(json.dumps(result))

    elif args.cmd == "validate":
        with open(args.coco_json) as f:
            coco = json.load(f)
        issues = validate_annotations(coco.get("annotations", []), bbox_format=args.bbox_fmt)
        if issues:
            for iss in issues:
                print(f"[{iss['level'].upper()}] {iss['code']}: {iss['message']}")
        else:
            print("All annotations valid.")

    elif args.cmd == "stats":
        with open(args.coco_json) as f:
            coco = json.load(f)
        stats = compute_stats(coco, bbox_format=args.bbox_fmt, top=args.top)
        print(json.dumps(stats, indent=2))
