"""Smoke guard for the HF Jobs scripts in ``jobs/``.

These run remotely via ``hf jobs uv run`` and clone this repo for the shared
``tools/`` + ``workflows/`` helpers, so a syntax error or a malformed PEP-723
header only surfaces *after* a job is submitted and billed. This test catches
both locally, with no network — the cheap proxy for "the Jobs pipeline still
launches".
"""

import ast
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

JOBS_DIR = Path(__file__).resolve().parents[2] / "jobs"
JOB_SCRIPTS = sorted(JOBS_DIR.glob("*.py"))


def _pep723_block(text: str) -> str | None:
    """Return the inline-script-metadata TOML body, or None if absent."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "# /// script")
    except StopIteration:
        return None
    body = []
    for ln in lines[start + 1:]:
        if ln.strip() == "# ///":
            return "\n".join(body)
        # strip the leading "# " (or bare "#") comment marker
        body.append(ln[2:] if ln.startswith("# ") else ln.lstrip("#"))
    raise AssertionError("PEP-723 block opened with '# /// script' but never closed")


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_job_script_compiles(script):
    ast.parse(script.read_text(), filename=str(script))


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_pep723_header_is_valid_when_present(script):
    block = _pep723_block(script.read_text())
    if block is None:
        pytest.skip(f"{script.name} is not a standalone uv script")
    meta = tomllib.loads(block)
    assert "dependencies" in meta, f"{script.name} PEP-723 header has no dependencies"
    assert isinstance(meta["dependencies"], list)


def test_at_least_the_core_pipeline_scripts_are_uv_scripts():
    """The label/judge/merge/train stages must each be launchable as uv scripts."""
    names = {p.name for p in JOB_SCRIPTS}
    for stage in ("label_qwen.py", "judge_one.py", "merge_judges.py", "train_rfdetr_job.py"):
        assert stage in names, f"expected pipeline stage {stage} in jobs/"
        block = _pep723_block((JOBS_DIR / stage).read_text())
        assert block is not None, f"{stage} must carry a PEP-723 uv header"


def _import_job(monkeypatch, name):
    monkeypatch.setenv("REPO_DIR", str(JOBS_DIR.parent))
    sys.modules.pop(f"jobs.{name}", None)
    return importlib.import_module(f"jobs.{name}")


def test_merge_job_defaults_to_higher_recall_policy(monkeypatch):
    module = _import_job(monkeypatch, "merge_judges")

    args = module.build_parser().parse_args(["--verdicts", "judge::verdicts.parquet"])

    assert args.min_agree == 1


def test_train_job_requires_artifact_ids_and_uses_detections(monkeypatch):
    module = _import_job(monkeypatch, "train_rfdetr_job")
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args([
        "--source", "owner/data-agree1",
        "--hub-model-id", "owner/model-agree1",
    ])

    assert args.annotation_source == "detections"


def test_train_job_forwards_annotation_source(monkeypatch):
    module = _import_job(monkeypatch, "train_rfdetr_job")
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "train_rfdetr_job.py",
        "--source", "owner/data-agree1",
        "--hub-model-id", "owner/model-agree1",
    ])

    module.main()

    command = captured["command"]
    option = command.index("--annotation-source")
    assert command[option + 1] == "detections"
