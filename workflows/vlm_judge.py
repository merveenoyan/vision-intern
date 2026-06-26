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

DEFAULT_VLM = "Qwen/Qwen3.5-9B"


def _is_local_dir(source: str | Path) -> bool:
    return Path(source).is_dir()


# ------------------------------------------------------------------
# Object specification  (a single "prompt-creation" meta-prompt)
# ------------------------------------------------------------------
#
# The judge no longer carries per-use-case category hints.  Instead, given the
# user's label set, the meta-prompt below has an LLM write one detailed positive
# definition per label — what the object looks like.  The resulting
# ``{label: definition}`` map is reused by BOTH the labeller (tools.vlm_detect
# ``class_descriptions``) and this judge, so both stages share one definition of
# each category without hand-written, dataset-specific dictionaries.


def build_object_spec_prompt(labels: list[str]) -> str:
    """Meta-prompt for *prompt creation*.

    Given the user's *labels*, instruct an LLM to write one detailed, positive
    visual definition per label.  Each definition must describe the **physical
    object a detector draws a box around** — naming the kind of object first and
    treating its text/symbols/colour as features carried *on* that object (e.g.
    "a traffic sign showing a black left arrow crossed by a red slash", not just
    "a black left arrow") — so the judge evaluates the whole object, not the
    bare symbol or concept.  Definitions are purely positive: no "not a …"
    negatives or look-alike lists.  The output feeds both the labelling prompt
    and the judge prompt.
    """
    label_block = "\n".join(f"  - {label}" for label in labels) or "  (none)"
    return (
        "You are writing the category definitions for an object-detection "
        "labelling and quality-control pipeline.\n"
        "These categories all come from ONE dataset.\n"
        f"Define every category in this set:\n{label_block}\n\n"
        "For EACH category, write ONE detailed, positive definition. Define the "
        "PHYSICAL OBJECT a detector draws a bounding box around: name the kind "
        "of object first, then describe the text, symbols, colour and shape it "
        "CARRIES as features ON that object — e.g. \"a traffic sign showing a "
        "black left arrow crossed by a red slash\", NOT just \"a black left "
        "arrow\". Never define the bare symbol, message, or concept in "
        "isolation. Expand any abbreviated or coded label name into plain "
        "words. When several categories are visually similar, make each "
        "definition specific enough to tell them apart — but describe only what "
        "the object IS, never what it is not.\n"
        "Return ONLY a JSON object mapping each category to its definition:\n"
        '{"<category>": "<definition>", ...}'
    )


def _parse_object_specs(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    return {str(k).lower(): str(v) for k, v in raw.items() if v}


def generate_object_specs(
    labels: list[str],
    model_id: str,
    *,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Run the prompt-creation meta-prompt and return ``{label: definition}``.

    The task is text-only; a small blank image satisfies the VLM client without
    needing a separate text endpoint.  (Some providers reject 1×1 images, so we
    use 64×64.)  Called once per run, not per image.

    Every label must end up with a non-empty definition: the router occasionally
    returns an empty 200 (no exception, so run_vlm's own retry doesn't fire) or
    silently omits/blanks individual categories, and a missing definition
    degrades that class to bare-label judging for the whole run. So we re-ask —
    targeting only the labels still missing — until all are filled or we run out
    of attempts, then print the full set for inspection.
    """
    if not labels:
        return {}
    wanted = [c.lower() for c in labels]
    blank = Image.new("RGB", (64, 64), "white")
    specs: dict[str, str] = {}
    max_attempts = 6
    for attempt in range(max_attempts):
        missing = [c for c in wanted if not specs.get(c)]
        if not missing:
            break
        try:
            response = run_vlm(
                blank, build_object_spec_prompt(missing), model_id,
                backend=backend, base_url=base_url, api_key=api_key,
                max_tokens=4096,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [object-spec] attempt {attempt + 1} raised: {e}", flush=True)
            continue
        for k, v in _parse_object_specs(response).items():
            if k in wanted and v and v.strip():
                specs[k] = v.strip()
        filled = sum(1 for c in wanted if specs.get(c))
        print(f"  [object-spec] attempt {attempt + 1}: {filled}/{len(wanted)} "
              f"definitions filled", flush=True)

    missing = [c for c in wanted if not specs.get(c)]
    if missing:
        print(f"  [warn] object-spec: {len(missing)} label(s) still undefined "
              f"after {max_attempts} attempts: {missing}", flush=True)

    # Print every created definition so the prompts are auditable before the run.
    print("  ===== category definitions =====", flush=True)
    for c in wanted:
        print(f"  - {c}: {specs.get(c, '(UNDEFINED)')}", flush=True)
    print("  ================================", flush=True)
    return specs


def _build_judge_prompt_overlay(
    detections: list[dict],
    class_descriptions: dict[str, str] | None = None,
) -> str:
    """Judge prompt for the *overlay* image: each box is drawn and numbered on
    the image itself, so we only list the proposed label per index — no raw
    coordinates for the model to mentally project."""
    lines = [
        f'  #{i}: proposed label "{det.get("label", "unknown")}"'
        for i, det in enumerate(detections)
    ]
    ann_block = "\n".join(lines) if lines else "  (none)"

    # Definitions only for the labels actually proposed on this image.
    class_descriptions = class_descriptions or {}
    present = [c for c in dict.fromkeys(
        str(det.get("label", "")).lower() for det in detections)
        if c in class_descriptions]
    defs_block = ""
    if present:
        defs = "\n".join(f"  - {c}: {class_descriptions[c]}" for c in present)
        defs_block = f"Category definitions:\n{defs}\n\n"

    return (
        "You are a quality-control judge for object-detection annotations.\n"
        "Each proposed detection is drawn on the image as a numbered coloured "
        "box (#0, #1, …) with its label.\n\n"
        f"{defs_block}"
        f"Proposed detections:\n{ann_block}\n\n"
        "Look at what is actually inside each numbered box and evaluate:\n"
        "  1. Is the labelled object actually present inside that box, matching "
        "the category definition above?\n"
        "  2. Is the box reasonably tight around the object?\n"
        "Return ONLY a JSON array — one entry per box:\n"
        '[{"id": <box number>, "verdict": "correct"|"incorrect"|"imprecise", '
        '"score": <0.0-1.0>, "reason": "<short explanation>"}]\n'
        "If there are no boxes return: []"
    )


def _build_judge_prompt_coco(
    annotations: list[dict],
    id2name: dict[int, str],
    width: int,
    height: int,
    class_descriptions: dict[str, str] | None = None,
) -> str:
    class_descriptions = class_descriptions or {}
    lines = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        label = id2name[ann["category_id"]]
        lines.append(
            f'  id={ann["id"]}: label="{label}", '
            f"bbox=[x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}]"
        )
    ann_block = "\n".join(lines) if lines else "  (none)"

    present = [c for c in dict.fromkeys(
        str(id2name[ann["category_id"]]).lower() for ann in annotations)
        if c in class_descriptions]
    defs_block = ""
    if present:
        defs = "\n".join(f"  - {c}: {class_descriptions[c]}" for c in present)
        defs_block = f"Category definitions:\n{defs}\n\n"

    return (
        "You are a quality-control judge for object-detection annotations.\n"
        f"Image size: {width}\u00d7{height} pixels.\n\n"
        f"{defs_block}"
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
        if not isinstance(item, dict):
            continue
        aid = item.get("id")
        if aid is None:
            continue
        # Judges sometimes echo the box label form ("#0", "box 0") rather than a
        # bare int — pull the first integer out of whatever they returned.
        if isinstance(aid, str):
            m = re.search(r"\d+", aid)
            if not m:
                continue
            aid = m.group()
        try:
            idx = int(aid)
        except (TypeError, ValueError):
            continue
        out[idx] = {
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
    overlay_img: Image.Image | None = None,
    class_descriptions: dict[str, str] | None = None,
) -> list[dict]:
    """Run a single judge over one image's detections.

    The judge sees the **numbered box-overlay** image — either *overlay_img*
    (the ``detections_overlay`` column produced at label time) or, if not
    supplied, one rendered on the fly — and is asked to evaluate each box by
    looking at it, rather than being handed raw coordinates over the bare
    image (which forced the model to project pixel coords it reasons about
    poorly).

    Returns one ``{detection_idx, verdict, score, reason}`` per detection.
    """
    if not detections:
        return []
    if overlay_img is None:
        from tools.bbox_viz import draw_detections
        overlay_img = draw_detections(img, detections, show_index=True)
    elif not isinstance(overlay_img, Image.Image):
        overlay_img = load_image(overlay_img)
    prompt = _build_judge_prompt_overlay(detections, class_descriptions)
    try:
        response = run_vlm(
            overlay_img, prompt, model_id,
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
    class_descriptions: dict[str, str] | None = None,
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
            class_descriptions,
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
    class_descriptions: dict[str, str] | None = None,
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

        overlay = row.get("detections_overlay")
        per_judge_row = {
            spec["model_id"]: score_detections(
                img, dets, spec["model_id"],
                backend=spec["backend"], base_url=spec["base_url"],
                api_key=spec["api_key"], overlay_img=overlay,
                class_descriptions=class_descriptions,
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
    labels: list[str] | None = None,
    class_descriptions: dict[str, str] | None = None,
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
    labels : list[str], optional
        The user's category set.  When given (and *class_descriptions* is not),
        the first judge runs the prompt-creation meta-prompt once to generate a
        visual definition per label, injected into the judge prompt to
        disambiguate look-alikes.  See :func:`build_object_spec_prompt`.
    class_descriptions : dict[str, str], optional
        Pre-built ``{label: definition}`` map.  Overrides *labels* generation;
        share it with the labeller (``tools.vlm_detect``) so both stages use
        the same definitions.
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

    # Build the category definitions once: use what the caller supplied, else
    # have the first judge generate them from the user's *labels* via the
    # prompt-creation meta-prompt.  Reused across every image/judge in the run.
    if class_descriptions is None and labels:
        first = judge_specs[0]
        class_descriptions = generate_object_specs(
            labels, first["model_id"],
            backend=first["backend"], base_url=first["base_url"],
            api_key=first["api_key"],
        )
        if class_descriptions:
            print(f"Generated definitions for {len(class_descriptions)} label(s): "
                  f"{sorted(class_descriptions)}")

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
            class_descriptions=class_descriptions,
        )
    return _judge_hub(
        str(source), str(output),
        judge_specs, threshold, min_agree,
        image_column, detections_column, split, max_samples,
        push_to_hub, hf_token, dataset_config=dataset_config,
        class_descriptions=class_descriptions,
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
    parser.add_argument("--labels", default=None,
                        help="Comma-separated category labels. The first judge "
                             "generates a visual definition per label (via the "
                             "prompt-creation meta-prompt) to disambiguate "
                             "look-alikes.")
    parser.add_argument("--class-descriptions", default=None,
                        help="JSON string or path to a {label: definition} map. "
                             "Overrides --labels generation.")
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

    label_list = (
        [c.strip() for c in args.labels.split(",") if c.strip()]
        if args.labels else None
    )

    class_descriptions = None
    if args.class_descriptions:
        spec = args.class_descriptions
        if Path(spec).is_file():
            spec = Path(spec).read_text()
        class_descriptions = {
            str(k).lower(): str(v) for k, v in json.loads(spec).items()
        }

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
        labels=label_list,
        class_descriptions=class_descriptions,
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
