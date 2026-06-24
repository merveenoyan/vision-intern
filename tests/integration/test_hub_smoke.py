"""Opt-in connectivity smoke test for the Hub / Jobs operator environment.

Skipped unless an HF token is available, so the default ``pytest`` run stays
offline and free. Run explicitly with::

    uv run --extra dev pytest -m integration

It does not submit a (billable) job — it confirms the machine is *able* to:
auth to the Hub, see the ``hf jobs`` CLI, and stream a public dataset (the data
path every labelling job depends on).
"""

import shutil

import pytest

pytestmark = pytest.mark.integration


def _has_token() -> bool:
    from huggingface_hub import get_token

    return get_token() is not None


@pytest.mark.skipif(not _has_token(), reason="no HF token; set HF_TOKEN or run `hf auth login`")
def test_hub_auth_works():
    from huggingface_hub import whoami

    info = whoami()
    assert info.get("name"), "whoami returned no user — token may be invalid"


def test_hf_cli_available():
    assert shutil.which("hf"), "`hf` CLI not on PATH — needed to launch HF Jobs"


@pytest.mark.skipif(not _has_token(), reason="no HF token")
def test_can_stream_a_public_dataset():
    """Pull a single row in streaming mode — the labelling data path, cheaply."""
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist", split="train", streaming=True)
    first = next(iter(ds))
    assert "image" in first
