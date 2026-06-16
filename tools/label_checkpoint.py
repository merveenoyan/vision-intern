"""Shared helpers for the decoupled local labelling pipeline.

Labelling and pushing are two separate scripts (:mod:`jobs.label_local` writes a
local checkpoint; :mod:`jobs.push_labeled` reads it and pushes to the Hub). Both
must key rows **identically** so the checkpoint lines up with the dataset — that
shared logic lives here.

The checkpoint is JSONL, one ``{"key": ..., "dets": [...]}`` record per labelled
image, appended atomically as each row finishes. Resuming = loading this file
and only labelling the keys that are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def load_deduped(source: str, *, dataset_config: str | None, split: str,
                 key_columns: list[str] | None, max_samples: int | None):
    """Load *source*, collapse repeated images, optionally cap — the exact same
    sequence both scripts run so their row order (and thus keys) match."""
    from datasets import load_dataset

    from tools.dataset_utils import dedupe_by_image

    ds = load_dataset(source, name=dataset_config, split=split)
    before = len(ds)
    ds = dedupe_by_image(ds, image_column="image", key_columns=key_columns or None)
    print(f"Deduped {before} → {len(ds)} unique images (key={key_columns})",
          flush=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def stable_keys(ds: "Dataset", key_columns: list[str] | None) -> list[str]:
    """One stable id per row — from *key_columns* (e.g. ``docId``) when present,
    else a content hash of the image. Identity is by id, not position, so the
    checkpoint survives ordering changes."""
    from tools.dataset_utils import image_key

    if key_columns and all(c in ds.column_names for c in key_columns):
        cols = {c: ds[c] for c in key_columns}
        return ["\x1f".join(str(cols[c][i]) for c in key_columns)
                for i in range(len(ds))]
    return [image_key(ds[i]["image"]) for i in range(len(ds))]


def load_checkpoint(path: "str | Path") -> dict[str, list[dict]]:
    """Read the JSONL checkpoint into ``{key: detections}``. Tolerates a
    truncated final line left by a hard crash mid-write."""
    path = Path(path)
    done: dict[str, list[dict]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line from a crash — skip
            done[rec["key"]] = rec["dets"]
    return done
