"""Atomic local labeller for ROAD SIGNS — STEP 1 of 2 (label only).

Road-sign variant of :mod:`jobs.label_local`: identical atomic/resumable
local-router machinery, but the class set and ``CLASS_DESCRIPTIONS`` are the 21
fine-grained traffic-sign categories of ``Francesco/road-signs-6ih4y``. Keep the
definitions here in sync with the judge hints in
:data:`workflows.vlm_judge._ROADSIGN_HINTS`.

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
# specific signs). Each definition excludes the look-alikes the VLM will
# otherwise confuse: directional no-turn signs vs each other, the three lit
# traffic-light colours vs the generic fixture, parking vs no-parking, the two
# pedestrian-crossing variants. Mirror in workflows.vlm_judge._ROADSIGN_HINTS.
ROADSIGN_DESCRIPTIONS = {
    "stop": ("a red octagonal sign with white 'STOP' text. NOT a generic red "
             "sign, a red circle, or a brake light."),
    "parking": ("a sign with a large letter 'P' meaning parking is allowed "
                "(typically white P on blue). NOT a 'no parking' sign that has "
                "a red slash or border."),
    "warning": ("a yellow or red triangular caution sign with a black symbol "
                "(general hazard). NOT a circular or rectangular sign."),
    "bus_stop": ("a sign marking a bus stop, usually showing a bus icon. NOT a "
                 "generic blue information sign."),
    "do_not_enter": ("a red circle crossed by a single horizontal white bar "
                     "(no entry / do not enter / wrong way). NOT a stop sign "
                     "and NOT a no-turn sign with an arrow."),
    "do_not_stop": ("a no-stopping sign: a circle (often blue or red) with a "
                    "red cross or a single red diagonal slash. NOT a plain 'P' "
                    "and NOT a no-parking sign."),
    "do_not_turn_l": ("a red circle with a LEFT-turn arrow crossed out (no left "
                      "turn). The arrow points left. NOT no-right-turn and NOT "
                      "no-U-turn."),
    "do_not_turn_r": ("a red circle with a RIGHT-turn arrow crossed out (no "
                      "right turn). The arrow points right. NOT no-left-turn "
                      "and NOT no-U-turn."),
    "do_not_u_turn": ("a red circle with a U-shaped turn arrow crossed out (no "
                      "U-turn). NOT a no-left-turn or no-right-turn sign."),
    "enter_left_lane": ("a sign directing traffic to enter or keep to the left "
                        "lane (a left/bent arrow, no crossing-out). NOT a "
                        "no-left-turn sign."),
    "green_light": ("a traffic signal head whose GREEN lamp is lit. NOT an "
                    "unlit signal, a red or yellow lamp, or a green street "
                    "lamp."),
    "left_right_lane": ("a sign showing that both left and right lanes/turns "
                        "are allowed (arrows pointing both left and right). NOT "
                        "a single-direction arrow."),
    "no_parking": ("a sign forbidding parking: a circle (often blue with a red "
                   "border and one red diagonal slash) or a 'P' crossed out. "
                   "NOT a plain 'P' parking-allowed sign."),
    "ped_crossing": ("a pedestrian-crossing WARNING sign: a triangle or diamond "
                     "showing a walking-person symbol. NOT the rectangular blue "
                     "zebra-crossing sign."),
    "ped_zebra_cross": ("a blue/white rectangular or square pedestrian-crossing "
                        "sign showing a person on striped (zebra) markings. NOT "
                        "the triangular warning version."),
    "railway_crossing": ("a level-crossing sign: an X-shaped (St Andrew's) "
                         "cross, or a triangle with a train/fence symbol. NOT a "
                         "generic warning triangle."),
    "red_light": ("a traffic signal head whose RED lamp is lit. NOT a yellow or "
                  "green lamp, a stop sign, or a tail light."),
    "t_intersection_l": ("a warning sign for a T-shaped intersection (the road "
                         "ends in a T). NOT a crossroads sign and NOT a "
                         "turn-arrow sign."),
    "traffic_light": ("a traffic signal head or 'signal ahead' sign showing the "
                      "light fixture where no single lamp colour is clearly lit "
                      "(or drawn as a symbol). NOT a single lit red/yellow/green "
                      "lamp."),
    "u_turn": ("a sign PERMITTING a U-turn: a U-shaped turn arrow with NO "
               "crossing-out. NOT a no-U-turn sign."),
    "yellow_light": ("a traffic signal head whose YELLOW/AMBER lamp is lit. NOT "
                     "a red or green lamp."),
}

DEFAULT_CLASSES = ",".join(ROADSIGN_DESCRIPTIONS.keys())


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

    token = os.environ["HF_TOKEN"]
    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    class_set = set(classes)
    class_descriptions = {c: ROADSIGN_DESCRIPTIONS[c] for c in classes
                          if c in ROADSIGN_DESCRIPTIONS}
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
