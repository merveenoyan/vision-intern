"""General Hugging Face ``datasets`` helpers for image datasets.

Two reusable utilities for the common case where a dataset has **multiple rows
sharing the same image** (e.g. VQA datasets with several question/answer rows
per page, or any source where one image yields many records):

- :func:`dedupe_by_image` — collapse to one row per unique image, so a
  labelling / annotation pass doesn't redundantly process (and pay for) the
  same image many times.
- :func:`grouped_train_val_split` — split into train/val without letting any
  image land in both halves, avoiding the leakage a plain row-wise
  ``Dataset.train_test_split`` causes on such datasets (inflated eval metrics).

Both key rows by an explicit column (fast, preferred when a stable id like
``docId`` exists) or, when none is given, by a content hash of the decoded
image pixels (works on any image dataset).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from datasets import Dataset
    from PIL import Image


def image_key(image: "str | Image.Image") -> str:
    """Return a stable content hash for *image* (md5 of its raw RGB bytes).

    Accepts a PIL Image, path, or URL. Two rows with pixel-identical images get
    the same key, which is what lets the dedupe / split helpers group repeated
    images without relying on a metadata id column.
    """
    from PIL import Image as _Image

    if not isinstance(image, _Image.Image):
        from tools.utils import load_image
        image = load_image(image)
    img = image.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def _row_keys(
    ds: "Dataset",
    key_columns: "str | Sequence[str] | None",
    image_column: str,
) -> list[str]:
    """One group key per row — from *key_columns* if given, else image hash."""
    if key_columns:
        cols = [key_columns] if isinstance(key_columns, str) else list(key_columns)
        missing = [c for c in cols if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"key_columns {missing} not in dataset (have {ds.column_names})"
            )
        columns = {c: ds[c] for c in cols}
        return ["\x1f".join(str(columns[c][i]) for c in cols) for i in range(len(ds))]

    if image_column not in ds.column_names:
        raise ValueError(
            f"image_column '{image_column}' not found and no key_columns given "
            f"(have {ds.column_names})"
        )
    # Decode one image at a time to avoid materialising the whole column.
    return [image_key(ds[i][image_column]) for i in range(len(ds))]


def dedupe_by_image(
    ds: "Dataset",
    *,
    image_column: str = "image",
    key_columns: "str | Sequence[str] | None" = None,
    keep: str = "first",
) -> "Dataset":
    """Reduce *ds* to one row per unique image.

    Parameters
    ----------
    ds : datasets.Dataset
        Dataset to deduplicate.
    image_column : str
        Image column, used when *key_columns* is not given.
    key_columns : str or sequence of str, optional
        Column(s) identifying the same image (e.g. ``"docId"`` or
        ``["doc_id", "page"]``). Preferred when available — far cheaper than
        hashing pixels. When omitted, rows are grouped by image content hash.
    keep : ``"first"`` | ``"last"``
        Which occurrence of each key to keep. Original row order is preserved.

    Returns
    -------
    datasets.Dataset
        Subset of *ds* with duplicate images removed.
    """
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")
    keys = _row_keys(ds, key_columns, image_column)
    chosen: dict[str, int] = {}
    for i, k in enumerate(keys):
        if keep == "first":
            chosen.setdefault(k, i)
        else:
            chosen[k] = i
    return ds.select(sorted(chosen.values()))


def grouped_train_val_split(
    ds: "Dataset",
    *,
    val_size: float = 0.15,
    group_columns: "str | Sequence[str] | None" = None,
    image_column: str = "image",
    seed: int = 42,
) -> "tuple[Dataset, Dataset | None]":
    """Split *ds* into ``(train, val)`` with no image spanning both splits.

    Unlike ``Dataset.train_test_split`` (which splits by row), this groups rows
    by image first — so a dataset with many rows per image can't leak the same
    image into both train and validation, which would inflate eval metrics.

    Parameters
    ----------
    ds : datasets.Dataset
        Dataset to split.
    val_size : float
        Target fraction of **groups** (≈ unique images) placed in validation.
        ``0`` returns ``(ds, None)``.
    group_columns : str or sequence of str, optional
        Column(s) identifying the same image. When omitted, groups by image
        content hash via *image_column*.
    image_column : str
        Image column, used when *group_columns* is not given.
    seed : int
        Seed for the deterministic group shuffle.

    Returns
    -------
    (datasets.Dataset, datasets.Dataset | None)
        Train and validation subsets. Validation is ``None`` when *val_size*
        is ``0`` or the dataset is too small to spare a group.
    """
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")
    if val_size == 0:
        return ds, None

    import random

    keys = _row_keys(ds, group_columns, image_column)
    groups = sorted(set(keys))
    random.Random(seed).shuffle(groups)

    n_val = round(len(groups) * val_size)
    if n_val == 0 or n_val >= len(groups):
        # Too few groups to carve a non-empty split on both sides.
        return ds, None

    val_groups = set(groups[:n_val])
    train_idx, val_idx = [], []
    for i, k in enumerate(keys):
        (val_idx if k in val_groups else train_idx).append(i)
    return ds.select(train_idx), ds.select(val_idx)
