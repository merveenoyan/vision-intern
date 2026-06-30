"""Smoke tests for the agent tool registry (offline, no model, no network).

These exercise the discovery + schema + dispatch layer in ``tools.registry``
and the role config in ``tools.config``.  Nothing here touches the network or
imports ``torch`` — the torch-heavy tools are only listed by name, never
materialised.
"""

import json
import subprocess
import sys

import pytest

from tools import config
from tools.registry import (
    _build_schema,
    _param_docs,
    as_json_schema,
    call,
    get_tool,
    get_tools,
    list_tools,
)

# Params that must never appear in an agent-facing schema.
BANNED = {"api_key", "base_url", "backend", "model_id"}


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


# ------------------------------------------------------------------
# Listing / filtering
# ------------------------------------------------------------------

def test_list_tools_train_filter_without_import():
    light = list_tools(include_train=False)
    full = list_tools(include_train=True)
    assert set(light) < set(full)
    # torch-heavy tools are listed only in the full set...
    assert "train" in full and "train" not in light
    assert "detect" in full and "detect" not in light
    # ...and the openai-backed VLM tools stay in the light set.
    assert {"vlm_detect", "label_dataset", "judge_labels"} <= set(light)


def test_default_get_tools_skips_train():
    names = {t.name for t in get_tools()}
    assert "train" not in names
    assert "vlm_detect" in names


# ------------------------------------------------------------------
# Schema validity
# ------------------------------------------------------------------

def test_every_light_tool_has_valid_json_schema():
    for spec in as_json_schema():
        # round-trips as JSON
        json.loads(json.dumps(spec))
        params = spec["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict) and params["properties"]
        required = params.get("required", [])
        assert set(required) <= set(params["properties"]), spec["name"]
        assert spec["description"]  # non-empty one-liner


def test_no_credential_params_leak():
    leaks = [
        (s["name"], k)
        for s in as_json_schema()
        for k in s["parameters"]["properties"]
        if k in BANNED
    ]
    assert leaks == []


def test_required_reflects_defaults():
    vd = get_tool("vlm_detect").parameters
    assert vd["required"] == ["image"]  # only the no-default, non-hidden param

    cb = get_tool("convert_bbox").parameters
    assert set(cb["required"]) == {"bbox", "from_fmt", "to_fmt"}


def test_param_types_mapped():
    props = get_tool("vlm_detect").parameters["properties"]
    assert props["image"]["type"] == "string"            # str | Image
    assert props["classes"] == {
        "type": "array", "items": {"type": "string"},
        "description": props["classes"]["description"],
    }
    cb = get_tool("convert_bbox").parameters["properties"]
    assert cb["bbox"]["type"] == "array"
    assert cb["img_w"]["type"] == "number"


def test_docstring_descriptions_attached():
    props = get_tool("vlm_detect").parameters["properties"]
    assert "Path" in props["image"]["description"]
    assert props["classes"]["description"]


def test_param_docs_parser_stops_at_returns():
    doc = (
        "Summary line.\n\n"
        "Parameters\n----------\n"
        "a : int\n    The first.\n"
        "b : str\n    The second,\n    continued.\n\n"
        "Returns\n-------\n"
        "int\n    not a param.\n"
    )
    parsed = _param_docs(doc)
    assert parsed == {"a": "The first.", "b": "The second, continued."}


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

def test_call_dispatches_cpu_tool():
    # xyxy [10,20,30,40] -> coco_xywh [10,20,20,20]
    assert call("convert_bbox", bbox=[10, 20, 30, 40],
                from_fmt="xyxy", to_fmt="coco_xywh") == [10, 20, 20, 20]


def test_call_unknown_tool():
    with pytest.raises(KeyError):
        call("does_not_exist")


def test_call_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        call("convert_bbox", bbox=[1, 2, 3, 4],
             from_fmt="xyxy", to_fmt="coco_xywh", bogus=1)


# ------------------------------------------------------------------
# Config injection (no network — the underlying fn is stubbed)
# ------------------------------------------------------------------

def test_call_injects_default_role_config(monkeypatch):
    spec = get_tool("vlm_detect")

    def fake(image=None, prompt=None, classes=None, model_id=None,
             backend=None, base_url=None, api_key=None, class_descriptions=None):
        return {"backend": backend, "model_id": model_id, "base_url": base_url}

    monkeypatch.setattr(spec, "fn", fake)
    config.configure(default=config.ToolConfig(
        backend="openai", model_id="M", base_url="http://u/v1"))

    assert call("vlm_detect", image="x.jpg") == {
        "backend": "openai", "model_id": "M", "base_url": "http://u/v1"}


def test_call_routes_judge_role(monkeypatch):
    spec = get_tool("judge_labels")
    captured = {}

    def fake(source=None, output=None, model_id=None, backend=None,
             base_url=None, api_key=None, **kw):
        captured.update(model_id=model_id, base_url=base_url)
        return captured

    monkeypatch.setattr(spec, "fn", fake)
    config.configure(judge=config.ToolConfig(
        model_id="judge-4b", base_url="http://judge/v1"))

    call("judge_labels", source="s", output="o")
    assert captured == {"model_id": "judge-4b", "base_url": "http://judge/v1"}


def test_explicit_kwarg_overrides_injected_config(monkeypatch):
    spec = get_tool("vlm_detect")

    def fake(image=None, model_id=None, backend=None, base_url=None,
             api_key=None, prompt=None, classes=None, class_descriptions=None):
        return model_id

    monkeypatch.setattr(spec, "fn", fake)
    config.configure(default=config.ToolConfig(model_id="from-config"))
    assert call("vlm_detect", image="x", model_id="explicit") == "explicit"


# ------------------------------------------------------------------
# The light path must not import torch
# ------------------------------------------------------------------

def test_default_path_does_not_import_torch():
    code = (
        "import sys; import tools; "
        "tools.as_json_schema(); tools.list_tools(); "
        "assert 'torch' not in sys.modules, 'torch leaked into the light path'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=".")


def test_build_schema_is_pure():
    # building a schema twice yields equal dicts (no hidden state mutation)
    a = _build_schema(get_tool("convert_bbox").fn, frozenset())
    b = _build_schema(get_tool("convert_bbox").fn, frozenset())
    assert a == b
