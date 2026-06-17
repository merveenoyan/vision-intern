"""Cut a single-split dataset into fixed, leak-free train/validation/test splits.

The trainer can split internally, but a *fixed* held-out test set makes eval
comparable across runs. This groups rows by image so no page spans two splits
(the leakage that inflates mAP on multi-row-per-image sources), carves test
first, then validation from the remainder, and pushes a ``DatasetDict``.

    HF_TOKEN=$(hf auth token) python3 jobs/make_splits.py \
        --source merve/docvqa-media3-judged-ensemble-trainval-agree1 \
        --output merve/docvqa-media3-judged-splits-agree1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Judged dataset repo")
    p.add_argument("--split", default="test", help="Input split to read")
    p.add_argument("--output", required=True, help="Destination repo (DatasetDict)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--group-columns", default="docId",
                   help="Column(s) grouping rows of the same page; '' = image hash.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from datasets import DatasetDict, load_dataset

    from tools.dataset_utils import grouped_train_val_split

    token = os.environ["HF_TOKEN"]
    group_cols = [c.strip() for c in args.group_columns.split(",") if c.strip()]
    group_cols = group_cols or None

    ds = load_dataset(args.source, split=args.split)
    print(f"Loaded {len(ds)} rows from {args.source}[{args.split}]", flush=True)

    # Carve the test set off first, then validation from what remains. The val
    # fraction is rescaled so it is val-frac of the WHOLE set, not the remainder.
    trainval, test = grouped_train_val_split(
        ds, val_size=args.test_frac, group_columns=group_cols,
        image_column="image", seed=args.seed)
    val_rel = args.val_frac / (1.0 - args.test_frac)
    train, val = grouped_train_val_split(
        trainval, val_size=val_rel, group_columns=group_cols,
        image_column="image", seed=args.seed)

    if test is None or val is None:
        raise SystemExit("Dataset too small to carve the requested splits.")

    dd = DatasetDict(train=train, validation=val, test=test)
    for name, part in dd.items():
        n_boxes = sum(len(d or []) for d in part["detections"])
        print(f"  {name}: {len(part)} pages / {n_boxes} boxes", flush=True)

    dd.push_to_hub(args.output, token=token)
    print(f"Pushed splits → https://huggingface.co/datasets/{args.output}")
    print("MAKE SPLITS DONE", flush=True)


if __name__ == "__main__":
    main()
