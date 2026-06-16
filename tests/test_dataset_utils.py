"""Tests for tools/dataset_utils.py.

Covers image_key, dedupe_by_image, and grouped_train_val_split using
datasets.Dataset.from_dict so no GPU is required.  The image_column path
(content-hash grouping) is tested with synthetic PIL Images; the key_columns
path avoids image hashing entirely so torch is never needed.
"""
from __future__ import annotations

import pytest
from datasets import Dataset as HFDataset
from PIL import Image as PILImage

from tools.dataset_utils import dedupe_by_image, grouped_train_val_split, image_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(r: int, g: int, b: int, size: tuple[int, int] = (8, 8)) -> PILImage.Image:
    return PILImage.new("RGB", size, (r, g, b))


def _ds(**cols) -> HFDataset:
    return HFDataset.from_dict(cols)


# ---------------------------------------------------------------------------
# image_key
# ---------------------------------------------------------------------------

class TestImageKey:
    def test_deterministic(self):
        img = _solid(100, 150, 200)
        assert image_key(img) == image_key(img)

    def test_identical_pixel_content_same_key(self):
        img1 = _solid(100, 150, 200)
        img2 = _solid(100, 150, 200)
        assert image_key(img1) == image_key(img2)

    def test_different_color_different_key(self):
        assert image_key(_solid(255, 0, 0)) != image_key(_solid(0, 255, 0))

    def test_different_size_different_key(self):
        img1 = _solid(100, 100, 100, size=(8, 8))
        img2 = _solid(100, 100, 100, size=(16, 16))
        assert image_key(img1) != image_key(img2)

    def test_returns_md5_hexdigest(self):
        key = image_key(_solid(10, 20, 30))
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_rgba_input_normalised_to_rgb(self):
        rgba = PILImage.new("RGBA", (4, 4), (100, 150, 200, 255))
        rgb = _solid(100, 150, 200, size=(4, 4))
        assert image_key(rgba) == image_key(rgb)


# ---------------------------------------------------------------------------
# dedupe_by_image  (key_columns path — no image hashing)
# ---------------------------------------------------------------------------

class TestDedupeByImage:
    def test_basic_key_columns(self):
        ds = _ds(doc_id=["a", "a", "b", "b", "c"], val=[1, 2, 3, 4, 5])
        deduped = dedupe_by_image(ds, key_columns="doc_id")
        assert len(deduped) == 3
        assert list(deduped["doc_id"]) == ["a", "b", "c"]
        assert list(deduped["val"]) == [1, 3, 5]  # first occurrence

    def test_keep_last(self):
        ds = _ds(doc_id=["a", "a", "b"], val=[10, 20, 30])
        deduped = dedupe_by_image(ds, key_columns="doc_id", keep="last")
        assert len(deduped) == 2
        assert list(deduped["val"]) == [20, 30]

    def test_keep_first_explicit(self):
        ds = _ds(doc_id=["x", "x"], val=[1, 2])
        deduped = dedupe_by_image(ds, key_columns="doc_id", keep="first")
        assert list(deduped["val"]) == [1]

    def test_no_duplicates_unchanged(self):
        ds = _ds(doc_id=["a", "b", "c"], val=[1, 2, 3])
        deduped = dedupe_by_image(ds, key_columns="doc_id")
        assert len(deduped) == 3

    def test_all_duplicates_keeps_one(self):
        ds = _ds(doc_id=["same"] * 5, val=list(range(5)))
        deduped = dedupe_by_image(ds, key_columns="doc_id")
        assert len(deduped) == 1

    def test_invalid_keep_raises(self):
        ds = _ds(doc_id=["a"], val=[1])
        with pytest.raises(ValueError, match="keep must be"):
            dedupe_by_image(ds, key_columns="doc_id", keep="random")

    def test_missing_key_column_raises(self):
        ds = _ds(doc_id=["a"], val=[1])
        with pytest.raises(ValueError):
            dedupe_by_image(ds, key_columns="nonexistent_col")

    def test_multi_column_key(self):
        ds = _ds(
            doc_id=["a", "a", "a", "b"],
            page=[1, 1, 2, 1],
            val=[10, 20, 30, 40],
        )
        # ("a",1) appears twice, ("a",2) once, ("b",1) once → 3 unique
        deduped = dedupe_by_image(ds, key_columns=["doc_id", "page"])
        assert len(deduped) == 3
        assert list(deduped["val"]) == [10, 30, 40]

    def test_preserves_row_order(self):
        ds = _ds(doc_id=["c", "a", "b", "a"], val=[3, 1, 2, 4])
        deduped = dedupe_by_image(ds, key_columns="doc_id", keep="first")
        # sorted(chosen.values()) → indices [0, 1, 2] → rows "c", "a", "b"
        assert list(deduped["doc_id"]) == ["c", "a", "b"]

    def test_image_column_path(self):
        img1 = _solid(255, 0, 0)
        img2 = _solid(0, 255, 0)
        ds = _ds(image=[img1, img1, img2], val=[1, 2, 3])
        deduped = dedupe_by_image(ds, image_column="image")
        assert len(deduped) == 2
        assert list(deduped["val"]) == [1, 3]


# ---------------------------------------------------------------------------
# grouped_train_val_split
# ---------------------------------------------------------------------------

def _multi_row_ds(n_groups: int, rows_per_group: int = 3) -> HFDataset:
    doc_ids = [str(g) for g in range(n_groups) for _ in range(rows_per_group)]
    xs = list(range(n_groups * rows_per_group))
    return _ds(doc_id=doc_ids, x=xs)


class TestGroupedTrainValSplit:
    def test_val_size_zero_returns_full_train(self):
        ds = _multi_row_ds(10)
        train, val = grouped_train_val_split(ds, val_size=0, group_columns="doc_id")
        assert val is None
        assert len(train) == len(ds)

    def test_too_few_groups_returns_full_train(self):
        # 2 groups × 0.15 → round(0.3) = 0 → (ds, None)
        ds = _multi_row_ds(2)
        train, val = grouped_train_val_split(ds, val_size=0.15, group_columns="doc_id")
        assert val is None
        assert len(train) == len(ds)

    def test_no_group_in_both_splits(self):
        ds = _multi_row_ds(10, rows_per_group=3)
        train, val = grouped_train_val_split(ds, val_size=0.3, group_columns="doc_id")
        assert val is not None
        train_groups = set(train["doc_id"])
        val_groups = set(val["doc_id"])
        assert train_groups.isdisjoint(val_groups)

    def test_all_rows_accounted_for(self):
        ds = _multi_row_ds(10, rows_per_group=3)
        train, val = grouped_train_val_split(ds, val_size=0.3, group_columns="doc_id")
        assert val is not None
        assert len(train) + len(val) == len(ds)

    def test_split_is_deterministic(self):
        ds = _multi_row_ds(20, rows_per_group=2)
        train1, val1 = grouped_train_val_split(ds, val_size=0.2, group_columns="doc_id", seed=7)
        train2, val2 = grouped_train_val_split(ds, val_size=0.2, group_columns="doc_id", seed=7)
        assert list(train1["x"]) == list(train2["x"])
        assert list(val1["x"]) == list(val2["x"])

    def test_different_seeds_may_differ(self):
        ds = _multi_row_ds(20, rows_per_group=2)
        _, val1 = grouped_train_val_split(ds, val_size=0.2, group_columns="doc_id", seed=1)
        _, val2 = grouped_train_val_split(ds, val_size=0.2, group_columns="doc_id", seed=999)
        assert set(val1["doc_id"]) != set(val2["doc_id"])

    def test_every_group_row_lands_in_same_split(self):
        ds = _multi_row_ds(10, rows_per_group=4)
        train, val = grouped_train_val_split(ds, val_size=0.3, group_columns="doc_id")
        assert val is not None
        val_groups = set(val["doc_id"])
        for row in val:
            assert row["doc_id"] in val_groups
        for row in train:
            assert row["doc_id"] not in val_groups

    def test_invalid_val_size_raises(self):
        ds = _multi_row_ds(5)
        with pytest.raises(ValueError):
            grouped_train_val_split(ds, val_size=1.5, group_columns="doc_id")

    def test_val_size_one_raises(self):
        ds = _multi_row_ds(5)
        with pytest.raises(ValueError):
            grouped_train_val_split(ds, val_size=1.0, group_columns="doc_id")

    def test_multi_column_group_key(self):
        ds = _ds(
            doc_id=["a", "a", "a", "b", "b", "b", "a", "a", "a", "b", "b", "b",
                    "c", "c", "c", "d", "d", "d", "e", "e", "e", "f", "f", "f"],
            page=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            x=list(range(24)),
        )
        train, val = grouped_train_val_split(
            ds, val_size=0.3, group_columns=["doc_id", "page"]
        )
        assert val is not None
        train_groups = set(zip(train["doc_id"], train["page"]))
        val_groups = set(zip(val["doc_id"], val["page"]))
        assert train_groups.isdisjoint(val_groups)
