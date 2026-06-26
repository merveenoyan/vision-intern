"""Generate judge category definitions via the prompt-creation meta-prompt.

This is the **one human-approval gate** in the e2e pipeline, kept deliberately
**separate** from :func:`workflows.judge_labels`.

Why separate: the abbreviated / coded label names a detection dataset ships with
(``do_not_turn_l``, ``t_intersection_l``, ``ped_zebra_cross``, …) mean little to
a VLM judge on their own. The meta-prompt
(:func:`workflows.vlm_judge.build_object_spec_prompt`) has an LLM expand each
label into a detailed, positive **visual definition**. Those definitions steer
*every* judge verdict, so a human should read — and, if needed, edit — them
**before any image is judged**, and nothing else in the run needs approval.

Flow
----
1. ``gen_descriptions`` runs the meta-prompt and writes ``{label: definition}``
   to a JSON file, then prints it for review.
2. A human reads / edits the JSON and approves it.
3. ``judge_labels(class_descriptions=<approved map>)`` (or
   ``vlm_judge --class-descriptions <file>``) consumes it **verbatim** and never
   regenerates — so the approved text is exactly what the judges see.

    # 1. generate (writes + prints definitions, then stops for review)
    uv run python -m workflows.gen_descriptions \
        --source merve/roadsign-labeled-qwen --drop road-signs \
        --backend openai --base-url https://router.huggingface.co/v1 \
        --model Qwen/Qwen3.5-9B --output descriptions.json

    # 2. human reviews / edits descriptions.json and approves

    # 3. judge with the approved file (no regeneration)
    uv run python -m workflows.vlm_judge --source merve/roadsign-labeled-qwen \
        --output merve/roadsign-judged-ensemble --push-to-hub \
        --class-descriptions descriptions.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflows.vlm_judge import generate_object_specs


def collect_labels(
    source: str,
    *,
    dataset_config: str | None = None,
    split: str = "train",
    detections_column: str = "detections",
    drop: tuple[str, ...] = (),
    scan_detections: bool = False,
    max_scan: int | None = None,
) -> list[str]:
    """Discover the label set of a Hugging Face detection dataset.

    Two strategies:

    * **ClassLabel names** (default, free — no image download): read the
      ``objects.category`` ClassLabel feature names. Datasets converted from the
      RF100 family carry the dataset's own name as a supercategory at index 0
      (e.g. ``road-signs``); pass it via *drop* to exclude it.
    * **Scan detections** (*scan_detections=True*): stream rows and collect the
      distinct ``detections[].label`` strings actually present — use this when
      the labeller emitted free-form labels that differ from the ClassLabel set.

    Either way the returned strings are what the judge matches against
    (lower-cased), so the generated definitions key correctly.
    """
    from datasets import load_dataset

    drop_lower = {d.strip().lower() for d in drop}

    if not scan_detections:
        ds = load_dataset(source, name=dataset_config, split=split, streaming=True)
        feats = ds.features or {}
        cat = None
        if "objects" in feats:
            try:
                cat = feats["objects"]["category"]
            except (TypeError, KeyError):
                cat = None
        names = getattr(getattr(cat, "feature", cat), "names", None)
        if names:
            return [n for n in names if n.lower() not in drop_lower]
        # No ClassLabel metadata — fall through to scanning detections.

    ds = load_dataset(source, name=dataset_config, split=split, streaming=True)
    labels: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        if max_scan is not None and i >= max_scan:
            break
        for det in row.get(detections_column) or []:
            lab = str(det.get("label", "")).strip()
            key = lab.lower()
            if lab and key not in seen and key not in drop_lower:
                seen.add(key)
                labels.append(lab)
    return labels


def gen_descriptions(
    labels: list[str],
    model_id: str,
    output: str | Path,
    *,
    backend: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Run the meta-prompt for *labels* and write ``{label: definition}`` to
    *output* (JSON) for human review. Returns the map.

    Does **not** judge anything and does **not** decide the map is good — that is
    the human's call. Any label the model left undefined is written as an empty
    string and flagged loudly so it is obvious what still needs a hand-written
    definition before approval.
    """
    if not labels:
        raise ValueError("No labels to define — pass --labels or a --source with "
                         "a ClassLabel `objects.category` / a `detections` column.")

    specs = generate_object_specs(
        labels, model_id, backend=backend, base_url=base_url, api_key=api_key,
    )
    # Keep every requested label as a key (empty string when undefined) so the
    # reviewer sees the full set and nothing is silently dropped.
    full = {lab.lower(): specs.get(lab.lower(), "") for lab in labels}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full, indent=2, ensure_ascii=False, sort_keys=True))

    undefined = [k for k, v in full.items() if not v]
    print("\n" + "=" * 70)
    print(f"Wrote {len(full)} judge definition(s) → {out_path}")
    if undefined:
        print(f"  !! {len(undefined)} label(s) UNDEFINED (empty) — fill before "
              f"approving: {undefined}")
    print("  REVIEW / EDIT this file, then pass it to the judge as "
          "--class-descriptions.")
    print("  The judge uses it verbatim and will NOT regenerate.")
    print("=" * 70 + "\n")
    return full


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Generate {label: definition} judge descriptions for review "
                    "(the one human-approval gate before judging).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="HF dataset to read the label set from")
    src.add_argument("--labels", help="Comma-separated labels (skips --source)")
    p.add_argument("--output", default="descriptions.json",
                   help="Where to write the {label: definition} JSON")
    p.add_argument("--drop", default="",
                   help="Comma-separated labels to exclude (e.g. an RF100 "
                        "supercategory like 'road-signs')")
    p.add_argument("--scan-detections", action="store_true",
                   help="Collect labels from the detections column instead of "
                        "the ClassLabel feature (for free-form labeller output)")
    p.add_argument("--detections-column", default="detections")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--max-scan", type=int, default=None,
                   help="Cap rows scanned when --scan-detections is set")
    p.add_argument("--model", required=True, help="Model that writes the definitions")
    p.add_argument("--backend", default="openai", choices=["openai", "transformers"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    drop = tuple(d.strip() for d in args.drop.split(",") if d.strip())
    if args.labels:
        labels = [c.strip() for c in args.labels.split(",") if c.strip()]
    else:
        labels = collect_labels(
            args.source, dataset_config=args.dataset_config, split=args.split,
            detections_column=args.detections_column, drop=drop,
            scan_detections=args.scan_detections, max_scan=args.max_scan,
        )
        print(f"Discovered {len(labels)} label(s) from {args.source}: {labels}")

    gen_descriptions(
        labels, args.model, args.output,
        backend=args.backend, base_url=args.base_url, api_key=args.api_key,
    )


if __name__ == "__main__":
    _cli()
