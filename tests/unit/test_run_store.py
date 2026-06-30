"""Offline tests for the run-artifact locator (no network, no torch).

Exercises the local ↔ bucket path resolution that lets a stage write to the
same logical location on HF Jobs (the ``/data`` mount) and locally (a bucket
``hf://`` URI or a plain directory).
"""

import pytest

from tools import run_store


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)


def test_explicit_data_root_wins_over_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", "/data")
    assert run_store.resolve_data_root(str(tmp_path)) == str(tmp_path)


def test_env_data_root_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "hf://buckets/me/runs")
    assert run_store.resolve_data_root() == "hf://buckets/me/runs"


def test_mount_used_when_present(monkeypatch, tmp_path):
    # Point the "mount" probe at a dir that exists → that root is used.
    assert run_store.resolve_data_root(mount=str(tmp_path)) == str(tmp_path)


def test_bucket_fallback_when_no_mount(monkeypatch):
    root = run_store.resolve_data_root(bucket="me/runs", mount="/definitely/not/here")
    assert root == "hf://buckets/me/runs"


def test_resolve_artifact_passes_through_uri_and_absolute():
    uri = "hf://buckets/me/runs/v.parquet"
    assert run_store.resolve_artifact(uri) == uri
    assert run_store.resolve_artifact("/data/run/v.parquet") == "/data/run/v.parquet"


def test_resolve_artifact_joins_relative(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "hf://buckets/me/runs")
    assert (run_store.resolve_artifact("roadsigns/v.parquet")
            == "hf://buckets/me/runs/roadsigns/v.parquet")


def test_resolve_artifact_strips_redundant_slashes(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data/")
    assert run_store.resolve_artifact("/run/v.parquet") == "/run/v.parquet"  # absolute → as-is
    assert run_store.resolve_artifact("run/v.parquet") == "/data/run/v.parquet"


def test_write_parquet_makes_local_parent_dirs(tmp_path):
    import pandas as pd

    target = tmp_path / "nested" / "dir" / "v.parquet"
    run_store.write_parquet(pd.DataFrame({"a": [1, 2]}), str(target))
    assert target.exists()
    assert run_store.read_parquet(str(target))["a"].tolist() == [1, 2]
