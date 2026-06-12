"""Evaluate and filter detection annotations using a VLM as a judge.

Accepts **local COCO annotations** or a **Hugging Face dataset** with a
``detections`` column produced by :func:`workflows.vlm_label.label_dataset`.

Local mode::

    python -m workflows.vlm_judge \\
        --source data/images --annotations annotations.json \\
        --output filtered.json

Hub mode::

    python -m workflows.vlm_judge \\
        --source merve/my-images-labeled \\
        --output merve/my-images-filtered --push-to-hub \\
        --backend openai --api-key hf_...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from tools.utils import load_image
from tools.vlm_client import run_vlm

DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


# ------------------------------------------------------------------
# Prompt / parsing (shared by both modes)
# ------------------------------------------------------------------

def _build_judge_prompt(
    detections: list[dict],
    width: int,
    height: int,
) -> str:
    lines = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox", det.get("box", []))
        label = det.get("label", "unknown")
        lines.append(f'  id={i}: label="{label}", bbox={bbox}')
    ann_block = "\n".join(lines) if lines else "  (none)"

    return (
        "You are a quality-control judge for object-detection annotations.\n"
        f"Image size: {width}\u00d7{height} pixels.\n\n"
        f"Current detections:\n{ann_block}\n\n"
        "For EACH detection evaluate:\n"
        "  1. Is the labelled object actually present at that location?\n"
        "  2. Is the bounding box reasonably tight around the object?\n"
        "Return ONLY a JSON array \u2014 one entry per detection:\n"
        '[{"id": <idx>, "verdict": "correct"|"incorrect"|"imprecise", '
        '"score": <0.0-1.0>, "reason": "<short explanation>"}]\n'
        "If there are no detections return: []"
    )


def _build_judge_prompt_coco(
    annotations: list[dict],
    id2name: dict[int, str],
    width: int,
    height: int,
) -> str:
    lines = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        label = id2name[ann["category_id"]]
        lines.append(
            f'  id={ann["id"]}: label="{label}", '
            f"bbox=[x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}]"
        )
    ann_block = "\n".join(lines) if lines else "  (none)"

    return (
        "You are a quality-control judge for object-detection annotations.\n"
        f"Image size: {width}\u00d7{height} pixels.\n\n"
        f"Current annotations:\n{ann_block}\n\n"
        "For EACH annotation evaluate:\n"
        "  1. Is the labelled object actually present at that location?\n"
        "  2. Is the bounding box reasonably tight around the object?\n"
        "Return ONLY a JSON array \u2014 one entry per annotation:\n"
        '[{"id": <ann_id>, "verdict": "correct"|"incorrect"|"imprecise", '
        '"score": <0.0-1.0>, "reason": "<short explanation>"}]\n'
        "If there are no annotations return: []"
    )


def _parse_verdicts(text: str) -> dict[int, dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    for item in raw:
        aid = item.get("id")
        if aid is None:
            continue
        out[int(aid)] = {
            "verdict": item.get("verdict", "incorrect"),
            "score": float(item.get("score", 0.0)),
            "reason": item.get("reason", ""),
        }
    return out


_NO_VERDICT = {"verdict": "incorrect", "score": 0.0, "reason": "no verdict"}


# ------------------------------------------------------------------
# Reusable scoring + ensemble  (shared by in-process multi-judge and
# the per-judge HF Jobs that write verdicts to a bucket)
# ------------------------------------------------------------------

def score_detections(
    img: Image.Image,
    detections: list[dict],
    model_id: str,
    *,
    backend: str,
    base_url: str | None,
    api_key: str | None,
) -> list[dict]:
    """Run a single judge over one image's detections.

    Returns one ``{detection_idx, verdict, score, reason}`` per detection.
    """
    if not detections:
        return []
    w, h = img.size
    prompt = _build_judge_prompt(detections, w, h)
    try:
        response = run_vlm(
            img, prompt, model_id,
            backend=backend, base_url=base_url, api_key=api_key,
            max_tokens=1024,
        )
    except Exception as e:  # noqa: BLE001 — a dead judge must not abort the run
        print(f"\n  [warn] judge {model_id} failed: {e}")
        response = "[]"
    verdicts = _parse_verdicts(response)
    return [
        {"detection_idx": i, **verdicts.get(i, _NO_VERDICT)}
        for i in range(len(detections))
    ]


def ensemble_row(
    per_judge_row: dict[str, list[dict]],
    n_dets: int,
    min_agree: int,
) -> list[dict]:
    """Combine several judges' per-detection verdicts for one image.

    *per_judge_row* maps a judge label → its list of
    ``{detection_idx, verdict, score, reason}`` dicts.  Returns one ensemble
    entry per detection with per-judge breakdown, vote count, mean score and
    an ``ensemble_keep`` flag (``n_correct >= min_agree``).
    """
    out: list[dict] = []
    for i in range(n_dets):
        per_judge: dict[str, dict] = {}
        scores: list[float] = []
        n_correct = 0
        for label, verdicts in per_judge_row.items():
            vmap = {v["detection_idx"]: v for v in verdicts}
            v = vmap.get(i, _NO_VERDICT)
            per_judge[label] = {
                "verdict": v.get("verdict", "incorrect"),
                "score": float(v.get("score", 0.0)),
                "reason": v.get("reason", ""),
            }
            scores.append(per_judge[label]["score"])
            if per_judge[label]["verdict"] == "correct":
                n_correct += 1
        mean = round(sum(scores) / len(scores), 4) if scores else 0.0
        out.append({
            "detection_idx": i,
            "per_judge": per_judge,
            "n_correct": n_correct,
            "mean_score": mean,
            "ensemble_keep": n_correct >= min_agree,
        })
    return out


def _normalize_judges(
    judges: list | None,
    model_id: str,
    backend: str,
    base_url: str | None,
    api_key: str | None,
) -> list[dict]:
    """Build a list of judge specs from either an explicit *judges* list
    (dicts or bare model-id strings) or the single-judge ``model_id`` args."""
    if not judges:
        return [{"model_id": model_id, "backend": backend,
                 "base_url": base_url, "api_key": api_key}]
    specs = []
    for j in judges:
        if isinstance(j, str):
            specs.append({"model_id": j, "backend": backend,
                          "base_url": base_url, "api_key": api_key})
        else:
            specs.append({
                "model_id": j["model_id"],
                "backend": j.get("backend", backend),
                "base_url": j.get("base_url", base_url),
                "api_key": j.get("api_key", api_key),
            })
    return specs


# ------------------------------------------------------------------
# Local mode  (image_dir + COCO JSON → filtered COCO JSON)
# ------------------------------------------------------------------

def _judge_local(
    image_dir: Path,
    annotations_path: Path,
    output_path: Path,
    model_id: str,
    threshold: float,
    backend: str,
    base_url: str | None,
    api_key: str | None,
) -> dict:
    with open(annotations_path) as f:
        coco = json.load(f)

    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    kept_anns: list[dict] = []
    verdicts_log: list[dict] = []

    for img_info in tqdm(coco["images"], desc="Judging"):
        anns = img_id_to_anns.get(img_info["id"], [])
        if not anns:
            continue

        image = load_image(image_dir / img_info["file_name"])
        prompt = _build_judge_prompt_coco(
            anns, id2name, img_info["width"], img_info["height"],
        )
        response = run_vlm(
            image, prompt, model_id,
            backend=backend, base_url=base_url, api_key=api_key,
            max_tokens=1024,
        )
        verdicts = _parse_verdicts(response)

        for ann in anns:
            v = verdicts.get(ann["id"], _NO_VERDICT)
            verdicts_log.append({"annotation_id": ann["id"], **v})
            if v["score"] >= threshold:
                kept_anns.append(ann)

    filtered = {
        "images": coco["images"],
        "annotations": kept_anns,
        "categories": coco["categories"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, indent=2))

    total = len(coco["annotations"])
    print(f"Judge kept {len(kept_anns)}/{total} annotations → {output_path}")

    verdicts_path = output_path.with_name(
        output_path.stem + "_verdicts" + output_path.suffix,
    )
    verdicts_path.write_text(json.dumps(verdicts_log, indent=2))
    return filtered


# ------------------------------------------------------------------
# Hub mode  (HF dataset with detections → filtered HF dataset)
# ------------------------------------------------------------------

def _judge_hub(
    dataset_id: str,
    output_id: str,
    judge_specs: list[dict],
    threshold: float,
    min_agree: int,
    image_column: str,
    detections_column: str,
    split: str,
    max_samples: int | None,
    push_to_hub: bool,
    hf_token: str | None,
    dataset_config: str | None = None,
) -> Any:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=dataset_config, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    labels = [s["model_id"] for s in judge_specs]
    print(f"Judging with {len(judge_specs)} judge(s): {labels} "
          f"(min_agree={min_agree}, threshold={threshold})")

    filtered_dets: list[list[dict]] = []
    all_verdicts: list[list[dict]] = []

    for row in tqdm(ds, desc="Judging", total=len(ds)):
        img = row[image_column]
        if not isinstance(img, Image.Image):
            img = load_image(img)

        dets = row.get(detections_column, []) or []
        if not dets:
            filtered_dets.append([])
            all_verdicts.append([])
            continue

        per_judge_row = {
            spec["model_id"]: score_detections(
                img, dets, spec["model_id"],
                backend=spec["backend"], base_url=spec["base_url"],
                api_key=spec["api_key"],
            )
            for spec in judge_specs
        }
        row_verdicts = ensemble_row(per_judge_row, len(dets), min_agree)

        kept = [
            det for det, v in zip(dets, row_verdicts)
            if v["ensemble_keep"] and v["mean_score"] >= threshold
        ]
        filtered_dets.append(kept)
        all_verdicts.append(row_verdicts)

    total = sum(len(row.get(detections_column, []) or []) for row in ds)

    if detections_column in ds.column_names:
        ds = ds.remove_columns([detections_column])
    ds = ds.add_column(detections_column, filtered_dets)
    ds = ds.add_column("judge_verdicts", all_verdicts)

    kept_total = sum(len(d) for d in filtered_dets)

    if push_to_hub:
        from tools.hub_viz import push_dataset_with_viz
        push_dataset_with_viz(
            ds, output_id, token=hf_token, image_column=image_column,
            detections_column=detections_column,
        )
        print(f"Judge kept {kept_total}/{total} detections "
              f"→ https://huggingface.co/datasets/{output_id}")
    else:
        ds.save_to_disk(output_id)
        print(f"Judge kept {kept_total}/{total} detections → {output_id}")

    return ds


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def judge_labels(
    source: str | Path,
    output: str | Path,
    annotations: str | Path | None = None,
    model_id: str = DEFAULT_VLM,
    judges: list | None = None,
    threshold: float = 0.0,
    min_agree: int = 1,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    image_column: str = "image",
    detections_column: str = "detections",
    split: str = "train",
    max_samples: int | None = None,
    push_to_hub: bool = False,
    hf_token: str | None = None,
    dataset_config: str | None = None,
) -> dict | Any:
    """Score annotations with a VLM judge and keep those above *threshold*.

    Parameters
    ----------
    source : str or Path
        **Local image directory** (requires *annotations*) or
        **HF dataset ID** with a ``detections`` column.
    output : str or Path
        Filtered COCO JSON path (local) or HF dataset ID (Hub).
    annotations : str or Path, optional
        COCO JSON file — required in local mode, ignored in Hub mode.
    model_id : str
        Single-judge model identifier (used when *judges* is not given).
    judges : list, optional
        Multiple judges for an **ensemble** (Hub mode).  Each entry is either
        a bare model-id string (inheriting *backend*/*base_url*/*api_key*) or a
        dict ``{"model_id", "backend", "base_url", "api_key"}``.  Per-judge
        verdicts plus the ensemble vote are stored in ``judge_verdicts``.
    threshold : float
        Minimum ensemble ``mean_score`` to keep a detection (0\u20131).
    min_agree : int
        Minimum number of judges that must vote ``correct`` to keep a
        detection.  ``0`` keeps everything (records verdicts only).
    backend / base_url / api_key
        Default inference backend for judges lacking their own spec.
    image_column / detections_column : str
        Column names (Hub mode only).
    split : str
        Dataset split (Hub mode only).
    max_samples : int, optional
        Cap the number of images to process.
    push_to_hub : bool
        Push the filtered dataset to the Hub (with a box-overlay gallery).
    hf_token : str, optional
        Hugging Face token for pushing.
    dataset_config : str, optional
        Dataset configuration name (Hub mode only).

    Returns
    -------
    dict or datasets.Dataset
        Filtered COCO dict (local) or HF Dataset (Hub).
    """
    judge_specs = _normalize_judges(judges, model_id, backend, base_url, api_key)

    if _is_local_dir(source):
        if annotations is None:
            raise ValueError(
                "Local mode requires --annotations (COCO JSON path)."
            )
        first = judge_specs[0]
        return _judge_local(
            Path(source), Path(annotations), Path(output),
            first["model_id"], threshold, first["backend"],
            first["base_url"], first["api_key"],
        )
    return _judge_hub(
        str(source), str(output),
        judge_specs, threshold, min_agree,
        image_column, detections_column, split, max_samples,
        push_to_hub, hf_token, dataset_config=dataset_config,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detection labels with a VLM judge",
    )
    parser.add_argument("--source", required=True,
                        help="Image directory or HF dataset ID")
    parser.add_argument("--output", required=True,
                        help="Filtered COCO JSON or HF dataset ID")
    parser.add_argument("--annotations", default=None,
                        help="COCO JSON (local mode only)")
    parser.add_argument("--model", default=DEFAULT_VLM)
    parser.add_argument("--judges", default=None,
                        help="Comma-separated judge model ids for an ensemble "
                             "(overrides --model). All share --backend/--base-url.")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-agree", type=int, default=1,
                        help="Min judges voting 'correct' to keep a detection "
                             "(0 = keep all, record verdicts only).")
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--detections-column", default="detections")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset config name (multi-config datasets)")
    args = parser.parse_args()

    judge_list = (
        [m.strip() for m in args.judges.split(",") if m.strip()]
        if args.judges else None
    )

    judge_labels(
        source=args.source,
        output=args.output,
        annotations=args.annotations,
        model_id=args.model,
        judges=judge_list,
        threshold=args.threshold,
        min_agree=args.min_agree,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        image_column=args.image_column,
        detections_column=args.detections_column,
        split=args.split,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    _cli()
