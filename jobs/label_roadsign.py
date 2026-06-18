"""Atomic local labeller for ROAD SIGNS — STEP 1 of 2 (label only).

Road-sign variant of :mod:`jobs.label_local`: identical atomic/resumable
local-router machinery, just preset to the 21 fine-grained traffic-sign
categories of ``Francesco/road-signs-6ih4y``. The disambiguating definitions are
generated from the class set at run time (see
:func:`workflows.vlm_judge.generate_object_specs`) — no hand-written dicts.

Runs the VLM detection calls **here** (router API requests, no GPU) and writes
each row's detections to a local JSONL checkpoint the instant they complete. It
does **not** touch the Hub — that's :mod:`jobs.push_labeled`.

- **Local + parallel**: router calls via a thread pool (default 16 workers).
- **Atomic**: each labelled row is appended + fsync'd immediately, keyed by a
  stable id (image content hash here — road frames have no doc-style id). A
  crash loses only in-flight rows.
- **Resume by default**: rows already in the checkpoint are skipped; only
  missing rows hit the router. ``--fresh`` ignores the checkpoint.

    HF_TOKEN=$(hf auth token) python3 jobs/label_roadsign.py \
        --checkpoint /tmp/roadsign-detections.jsonl

Then push with::

    python3 jobs/push_labeled.py \
        --source Francesco/road-signs-6ih4y --dataset-config default \
        --split train --dedupe-key-columns "" \
        --checkpoint /tmp/roadsign-detections.jsonl --output merve/roadsign-labeled-qwen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The 21 fine-grained road-sign classes of Francesco/road-signs-6ih4y (the
# dataset's "road-signs" super-category at index 0 is dropped — we detect the
# specific signs). This is just the dataset's label set, like the default
# --source; the disambiguating definitions are generated from it at run time.
DEFAULT_CLASSES = (
    "stop,parking,warning,bus_stop,do_not_enter,do_not_stop,do_not_turn_l,"
    "do_not_turn_r,do_not_u_turn,enter_left_lane,green_light,left_right_lane,"
    "no_parking,ped_crossing,ped_zebra_cross,railway_crossing,red_light,"
    "t_intersection_l,traffic_light,u_turn,yellow_light"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="Francesco/road-signs-6ih4y")
    p.add_argument("--dataset-config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--classes", default=DEFAULT_CLASSES)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--dedupe-key-columns", default="",
                   help="Empty → dedupe by image content hash (road frames "
                        "have no stable doc id).")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--checkpoint", default="/tmp/roadsign-detections.jsonl",
                   help="JSONL of {key,dets}; appended atomically, resumed on rerun.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint and relabel everything.")
    args = p.parse_args()

    from PIL import Image

    from tools.label_checkpoint import load_checkpoint, load_deduped, stable_keys
    from tools.utils import load_image
    from tools.vlm_detect import vlm_detect
    from workflows.vlm_judge import generate_object_specs

    token = os.environ["HF_TOKEN"]
    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    class_set = set(classes)
    # Disambiguating definitions generated once from the class set via the
    # prompt-creation meta-prompt (same generator the judge uses), then reused
    # for every row — no hand-written, dataset-specific dicts.
    class_descriptions = generate_object_specs(
        classes, args.model, backend="openai", base_url=None, api_key=token,
    )
    print(f"Generated definitions for {len(class_descriptions)}/{len(classes)} "
          f"classes: {sorted(class_descriptions)}", flush=True)
    key_cols = [c.strip() for c in args.dedupe_key_columns.split(",") if c.strip()]
    ckpt = Path(args.checkpoint)

    ds = load_deduped(args.source, dataset_config=args.dataset_config,
                      split=args.split, key_columns=key_cols or None,
                      max_samples=args.max_samples)
    keys = stable_keys(ds, key_cols or None)

    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"--fresh: removed {ckpt}", flush=True)

    done = load_checkpoint(ckpt)
    todo = [i for i, k in enumerate(keys) if k not in done]
    print(f"Checkpoint has {len(done)} rows; {len(todo)} of {len(ds)} still to "
          f"label (classes={classes}, model={args.model})", flush=True)
    if not todo:
        print("Nothing to label — checkpoint complete. LABEL DONE", flush=True)
        return

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    fh = ckpt.open("a", encoding="utf-8")

    def _label(i: int) -> int:
        img = ds[i]["image"]
        if not isinstance(img, Image.Image):
            img = load_image(img)
        try:
            dets = vlm_detect(
                img, classes=classes, model_id=args.model,
                backend="openai", base_url=None, api_key=token,
                class_descriptions=class_descriptions,
            )
            dets = [d for d in dets if d.get("label", "").lower() in class_set]
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] row {i} ({keys[i]}) failed: {e}", flush=True)
            return -1  # don't checkpoint failures — they retry next run
        with write_lock:
            fh.write(json.dumps({"key": keys[i], "dets": dets}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return i

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_label, i) for i in todo]):
            if fut.result() >= 0:
                completed += 1
                if completed % 50 == 0:
                    print(f"  [{completed}/{len(todo)}] labelled", flush=True)
    fh.close()

    done = load_checkpoint(ckpt)
    n_boxes = sum(len(v) for v in done.values())
    missing = [k for k in keys if k not in done]
    print(f"Labelled {completed} new rows this run; checkpoint now {len(done)} "
          f"rows / {n_boxes} boxes ({len(missing)} still missing)", flush=True)
    print(f"Next: python3 jobs/push_labeled.py --checkpoint {ckpt} --output <repo>")
    print("LABEL DONE", flush=True)


if __name__ == "__main__":
    main()
