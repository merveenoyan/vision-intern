"""Drop the inherited human-GT `objects` column from a judged dataset so the
RF-DETR trainer trains on our VLM `detections` (the trainer prioritizes
`objects` when present). Also drops the heavy `detections_overlay`. Non-
destructive: writes to a new `<src>-trainready` repo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2]
    from datasets import load_dataset

    token = os.environ["HF_TOKEN"]
    ds = load_dataset(src, split="train")
    drop = [c for c in ("objects", "detections_overlay") if c in ds.column_names]
    ds = ds.remove_columns(drop)
    n_boxes = sum(len(d or []) for d in ds["detections"])
    print(f"{src}: dropped {drop}; {len(ds)} rows / {n_boxes} detection boxes "
          f"→ cols now {ds.column_names}", flush=True)
    ds.push_to_hub(out, token=token, split="train")
    print(f"PUSHED {out}", flush=True)


if __name__ == "__main__":
    main()
