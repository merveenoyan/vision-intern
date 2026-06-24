"""Smoke tests for the dataset grouping helpers.

These use an in-memory ``datasets.Dataset`` keyed by an explicit id column, so
no images are decoded and nothing touches the network.
"""

import pytest

datasets = pytest.importorskip("datasets")

from tools.dataset_utils import dedupe_by_image, grouped_train_val_split  # noqa: E402


def _toy_dataset():
    # 3 unique "documents", several rows each (the VQA-style repeated-image case).
    return datasets.Dataset.from_dict(
        {
            "docId": ["a", "a", "a", "b", "b", "c"],
            "question": [f"q{i}" for i in range(6)],
        }
    )


def test_dedupe_keeps_one_row_per_key():
    ds = _toy_dataset()
    out = dedupe_by_image(ds, key_columns="docId")
    assert len(out) == 3
    assert sorted(out["docId"]) == ["a", "b", "c"]


def test_dedupe_keep_last_picks_last_occurrence():
    ds = _toy_dataset()
    first = dedupe_by_image(ds, key_columns="docId", keep="first")
    last = dedupe_by_image(ds, key_columns="docId", keep="last")
    # "a" first appears at row 0 and last at row 2
    assert first["question"][0] == "q0"
    assert last["question"][0] == "q2"


def test_dedupe_rejects_missing_key_column():
    with pytest.raises(ValueError):
        dedupe_by_image(_toy_dataset(), key_columns="nope")


def test_grouped_split_does_not_leak_a_group_across_halves():
    ds = _toy_dataset()
    train, val = grouped_train_val_split(ds, val_size=0.34, group_columns="docId", seed=0)
    assert val is not None
    train_docs, val_docs = set(train["docId"]), set(val["docId"])
    assert train_docs.isdisjoint(val_docs), "a docId leaked across train/val"
    assert train_docs | val_docs == {"a", "b", "c"}


def test_grouped_split_zero_val_returns_none():
    train, val = grouped_train_val_split(_toy_dataset(), val_size=0.0, group_columns="docId")
    assert val is None and len(train) == 6
