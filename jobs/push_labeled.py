"""Push a locally-labelled checkpoint to the Hub — STEP 2 of 2 (no router calls).

Reads the JSONL checkpoint written by :mod:`jobs.label_local`, reassembles the
``detections`` column in dataset order, renders the numbered box-overlay column
the judges score against, and pushes with the viz gallery. Fully decoupled from
labelling: rerun freely to retry a failed push without re-paying for any
detection call.

    HF_TOKEN=$(hf auth token) python3 jobs/push_labeled.py \
        --checkpoint /tmp/docvqa-media3-detections.jsonl \
        --output merve/docvqa-media3-labeled-qwen
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="lmms-lab/DocVQA")
    p.add_argument("--dataset-config", default="DocVQA")
    p.add_argument("--split", default="test")
    p.add_argument("--dedupe-key-columns", default="docId")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--checkpoint", default="/tmp/docvqa-media3-detections.jsonl")
    p.add_argument("--output", required=True, help="Destination HF dataset repo")
    p.add_argument("--allow-missing", action="store_true",
                   help="Push even if some rows have no checkpointed detections "
                        "(default: refuse, so you don't ship a half-labelled set).")
    p.add_argument("--drop-missing", action="store_true",
                   help="Drop rows with no checkpointed detections from the output "
                        "entirely (vs --allow-missing, which ships them as empty). "
                        "Use when unlabelled rows would be false negatives.")
    args = p.parse_args()

    from datasets import Image as HFImage
    from PIL import Image

    from tools.bbox_viz import draw_detections
    from tools.hub_viz import push_dataset_with_viz
    from tools.label_checkpoint import load_checkpoint, load_deduped, stable_keys
    from tools.utils import load_image

    token = os.environ["HF_TOKEN"]
    key_cols = [c.strip() for c in args.dedupe_key_columns.split(",") if c.strip()]

    ds = load_deduped(args.source, dataset_config=args.dataset_config,
                      split=args.split, key_columns=key_cols or None,
                      max_samples=args.max_samples)
    keys = stable_keys(ds, key_cols or None)
    done = load_checkpoint(args.checkpoint)

    missing = [k for k in keys if k not in done]
    if missing and args.drop_missing:
        keep_idx = [i for i, k in enumerate(keys) if k in done]
        ds = ds.select(keep_idx)
        keys = [keys[i] for i in keep_idx]
        print(f"Dropping {len(missing)} unlabelled rows; pushing {len(keys)} "
              f"labelled rows", flush=True)
    elif missing and not args.allow_missing:
        raise SystemExit(
            f"Refusing to push: {len(missing)}/{len(ds)} rows have no detections "
            f"in {args.checkpoint}. Finish labelling (rerun jobs/label_local.py), "
            f"pass --drop-missing to exclude them, or --allow-missing to push "
            f"them as empty.")
    elif missing:
        print(f"WARNING: pushing {len(missing)} rows with empty detections",
              flush=True)

    all_dets = [done.get(k, []) for k in keys]
    n_boxes = sum(len(d) for d in all_dets)
    print(f"Assembling {len(ds)} rows / {n_boxes} boxes → {args.output}", flush=True)

    # Encode overlays as PNG-bytes dicts — add_column can't infer an Arrow type
    # from raw PIL.Image objects, so build the {bytes,path} form HFImage casts.
    def _enc(img: Image.Image) -> dict:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return {"bytes": buf.getvalue(), "path": None}

    overlays = []
    for i in range(len(ds)):
        img = ds[i]["image"]
        if not isinstance(img, Image.Image):
            img = load_image(img)
        overlays.append(_enc(draw_detections(img, all_dets[i], show_index=True)))

    ds = ds.add_column("detections", all_dets)
    ds = ds.add_column("detections_overlay", overlays)
    ds = ds.cast_column("detections_overlay", HFImage())

    push_dataset_with_viz(ds, args.output, token=token, image_column="image",
                          detections_column="detections")
    print("PUSH DONE", flush=True)


if __name__ == "__main__":
    main()
