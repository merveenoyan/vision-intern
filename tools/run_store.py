"""Locate + read/write the run artifacts that pass between pipeline stages.

The judge → merge handoff writes a per-judge *verdicts* parquet that a later
stage reads back. That artifact's canonical home is a **bucket**, but the same
logical path has to work in two places:

* **On HF Jobs** the bucket is FUSE-mounted at ``/data`` — fast, local-looking IO.
* **Locally** there is no mount, so the bucket is addressed directly through the
  ``hf://buckets/<id>/…`` filesystem (``fsspec`` + ``huggingface_hub``, which
  resolves bucket URIs and picks up the HF token from the env / cache). A plain
  local directory also works, for a fully offline run.

So a stage never hard-codes ``/data``. It takes a *relative* artifact name (e.g.
``roadsigns/verdicts_gemma.parquet``) and lets :func:`resolve_artifact` place it:

    resolve_data_root() precedence:
      explicit data_root  →  $DATA_ROOT  →  /data (if mounted)  →  hf://buckets/<bucket>

An artifact path that is already absolute or a URI is passed through unchanged,
so the old fully-qualified ``--out /data/...`` / ``hf://buckets/...`` invocations
keep working. Reads/writes go through ``pandas`` (→ ``fsspec``), and we only
create parent directories for real local paths (never for a URI).

This module is CPU-only and imports no torch — safe in the light path.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BUCKET = "merve/vision-agent-runs"
MOUNT = "/data"


def _is_uri(p: str) -> bool:
    return "://" in p


def resolve_data_root(
    data_root: str | None = None,
    *,
    bucket: str = DEFAULT_BUCKET,
    mount: str = MOUNT,
) -> str:
    """Return the root under which run artifacts live.

    Precedence: explicit *data_root* → ``DATA_ROOT`` env → the ``/data`` bucket
    mount if it exists (i.e. running on a Job) → the bucket addressed directly
    over ``hf://`` (i.e. running locally).

    If ``/data`` happens to exist on your local box for unrelated reasons, pass
    *data_root* (or set ``DATA_ROOT``) to override the auto-detection.
    """
    root = data_root or os.environ.get("DATA_ROOT")
    if root:
        return root.rstrip("/")
    if Path(mount).is_dir():
        return mount
    return f"hf://buckets/{bucket}"


def resolve_artifact(
    name: str,
    *,
    data_root: str | None = None,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Resolve one artifact path.

    A *name* that is already absolute (``/data/...``) or a URI
    (``hf://buckets/...``) is returned unchanged; a relative *name* is joined
    onto :func:`resolve_data_root`.
    """
    if _is_uri(name) or os.path.isabs(name):
        return name
    root = resolve_data_root(data_root, bucket=bucket)
    return f"{root}/{name.strip('/')}"


def write_parquet(df, path: str) -> None:
    """Write *df* to *path* (local, ``/data`` mount, or ``hf://`` URI). Creates
    parent dirs only for real local paths — a URI manages its own namespace."""
    if not _is_uri(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def read_parquet(path: str):
    """Read a parquet from a local path, the ``/data`` mount, or an ``hf://`` URI."""
    import pandas as pd

    return pd.read_parquet(path)
