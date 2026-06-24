"""Combine per-split labelled checkpoints into ONE deduped labelled dataset.

The atomic labeller (:mod:`jobs.label_local`) is run once per split, each into
its own ``docId``-keyed checkpoint. This script stitches those splits back
together into a single labelled dataset for the judges to score:

1. Load each ``split::checkpoint`` pair exactly as the labeller did
   (``load_deduped`` collapses repeats within the split by ``docId``).
2. Attach that split's detections from its checkpoint, keyed identically.
3. Concatenate the splits, then **dedupe ACROSS the union by image-content
   hash** — a page that appears in both ``validation`` and ``test`` is kept
   once, so it can't later leak across our self-made train/val/test splits.
4. Render the numbered box overlays and push one labelled dataset.

Pure assembly — no router calls, so it is free to rerun after a failed push.

    HF_TOKEN=$(hf auth token) python3 jobs/build_combined_labeled.py \
        --input test::/tmp/docvqa-media3-test-dets.jsonl \
        --input validation::/tmp/docvqa-media3-val-dets.jsonl \
        --output merve/docvqa-media3-labeled-qwen-trainval
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
    p.add_argument("--input", action="append", required=True, dest="inputs",
                   help="Repeatable 'split::checkpoint_path' pair, one per split.")
    p.add_argument("--dedupe-key-columns", default="docId",
                   help="Per-split dedupe/key columns (must match the labeller).")
    p.add_argument("--output", required=True, help="Destination HF dataset repo")
    p.add_argument("--split-name", default="test",
                   help="Split name to push under (judges default to 'test').")
    p.add_argument("--allow-missing", action="store_true",
                   help="Push even if some rows have no checkpointed detections.")
    args = p.parse_args()

    from datasets import Image as HFImage
    from datasets import concatenate_datasets
    from PIL import Image

    from tools.bbox_viz import draw_detections
    from tools.dataset_utils import dedupe_by_image
    from tools.hub_viz import push_dataset_with_viz
    from tools.label_checkpoint import load_checkpoint, load_deduped, stable_keys
    from tools.utils import load_image

    token = os.environ["HF_TOKEN"]
    key_cols = [c.strip() for c in args.dedupe_key_columns.split(",") if c.strip()]

    parts = []
    for spec in args.inputs:
        split, ckpt = spec.split("::", 1)
        ds = load_deduped(args.source, dataset_config=args.dataset_config,
                          split=split, key_columns=key_cols or None,
                          max_samples=None)
        keys = stable_keys(ds, key_cols or None)
        done = load_checkpoint(ckpt)
        missing = [k for k in keys if k not in done]
        if missing and not args.allow_missing:
            raise SystemExit(
                f"Refusing to build: {len(missing)}/{len(ds)} rows in split "
                f"'{split}' have no detections in {ckpt}. Finish labelling "
                f"(rerun jobs/label_local.py --split {split}) or --allow-missing.")
        if missing:
            print(f"WARNING: split '{split}' has {len(missing)} empty rows",
                  flush=True)
        dets = [done.get(k, []) for k in keys]
        ds = ds.add_column("detections", dets)
        print(f"Split '{split}': {len(ds)} rows / {sum(len(d) for d in dets)} boxes",
              flush=True)
        parts.append(ds)

    combined = concatenate_datasets(parts)
    before = len(combined)
    # Cross-split dedupe by pixel content — never collides across splits the way
    # docId could, so a duplicated page survives in exactly one row.
    combined = dedupe_by_image(combined, image_column="image", key_columns=None)
    n_boxes = sum(len(d) for d in combined["detections"])
    print(f"Combined {before} → {len(combined)} unique pages / {n_boxes} boxes "
          f"→ {args.output}", flush=True)

    # Encode overlays as PNG-bytes dicts — add_column can't infer an Arrow type
    # from raw PIL.Image objects, so build the {bytes,path} form HFImage casts.
    def _enc(img: Image.Image) -> dict:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return {"bytes": buf.getvalue(), "path": None}

    overlays = []
    dets_col = combined["detections"]
    for i in range(len(combined)):
        img = combined[i]["image"]
        if not isinstance(img, Image.Image):
            img = load_image(img)
        overlays.append(_enc(draw_detections(img, dets_col[i], show_index=True)))

    combined = combined.add_column("detections_overlay", overlays)
    combined = combined.cast_column("detections_overlay", HFImage())

    push_dataset_with_viz(combined, args.output, token=token, image_column="image",
                          detections_column="detections", split=args.split_name)
    print("BUILD COMBINED DONE", flush=True)


if __name__ == "__main__":
    main()
