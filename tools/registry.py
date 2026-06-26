"""Agent-callable tool registry for vision-agent.

A thin discovery + invocation layer over the existing ``tools`` and
``workflows`` functions — it does **not** rewrite them.  An in-process
agent can:

>>> from tools import get_tools, as_json_schema, call
>>> [t.name for t in get_tools()]            # tools available without torch
>>> as_json_schema()[0]                      # framework-agnostic JSON Schema
>>> call("convert_bbox", bbox=[1, 2, 3, 4], from_fmt="xyxy", to_fmt="coco_xywh")

Each :class:`ToolSpec` carries a JSON-Schema description of its *task*
parameters, built by introspecting the underlying function's signature and
NumPy-style docstring.  Credential / endpoint params (``api_key``,
``base_url``, ``backend``, ``model_id``) are **hidden** from the schema and
filled at call time from :mod:`tools.config`, so an agent never sees — or is
asked to supply — an API key.

Specs are materialised lazily: ``get_tools()`` (the default) skips the
torch-heavy tools, so importing this module and listing tools on a CPU-only
install never imports ``torch``.
"""

from __future__ import annotations

import importlib
import inspect
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable

from .config import config_field_names, current


@dataclass
class ToolSpec:
    """A single agent-callable tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema (object) describing the visible task params
    fn: Callable
    hidden: frozenset
    role: str
    requires_train: bool

    def as_dict(self) -> dict:
        """Framework-agnostic ``{name, description, parameters}`` spec."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ------------------------------------------------------------------
# The registry table — pure data, no imports happen here.
# `hidden` params are dropped from the schema; the config-field subset
# (backend/model_id/base_url/api_key) is injected from tools.config at call
# time.  `requires_train=True` marks tools that import torch (the `train`
# extra) and are therefore skipped by the default get_tools().
# ------------------------------------------------------------------

_VLM_HIDDEN = {"backend", "base_url", "api_key", "model_id"}

_ENTRIES: list[dict] = [
    # --- VLM tools (openai backend → no torch needed) ---
    {"name": "vlm_detect", "module": "tools.vlm_detect", "attr": "vlm_detect",
     "hidden": _VLM_HIDDEN, "role": "default", "requires_train": False},
    {"name": "document_ocr", "module": "tools.document_ocr", "attr": "document_ocr",
     "hidden": _VLM_HIDDEN, "role": "default", "requires_train": False},
    {"name": "ocr_judge", "module": "tools.ocr_judge", "attr": "ocr_judge",
     "hidden": _VLM_HIDDEN, "role": "default", "requires_train": False},
    # --- pipeline workflows ---
    {"name": "label_dataset", "module": "workflows.vlm_label", "attr": "label_dataset",
     "hidden": _VLM_HIDDEN | {"hf_token"}, "role": "labeller", "requires_train": False},
    {"name": "judge_labels", "module": "workflows.vlm_judge", "attr": "judge_labels",
     "hidden": _VLM_HIDDEN | {"hf_token", "judges"}, "role": "judge",
     "requires_train": False},
    {"name": "train", "module": "workflows.train_rfdetr", "attr": "train",
     "hidden": set(), "role": "default", "requires_train": True},
    # --- local-GPU model tools (need the `train` extra / torch) ---
    {"name": "detect", "module": "tools.detect", "attr": "detect",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "grounded_detect", "module": "tools.grounded_detect", "attr": "grounded_detect",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "instance_segment", "module": "tools.instance_segment", "attr": "instance_segment",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "segment_from_bbox", "module": "tools.segment_from_bbox", "attr": "segment_from_bbox",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "segment_from_text", "module": "tools.segment_from_text", "attr": "segment_from_text",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "fast_segment", "module": "tools.fast_segment", "attr": "fast_segment",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "estimate_depth", "module": "tools.depth", "attr": "estimate_depth",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "estimate_pose", "module": "tools.pose", "attr": "estimate_pose",
     "hidden": set(), "role": "default", "requires_train": True},
    {"name": "ocr", "module": "tools.ocr", "attr": "ocr",
     "hidden": set(), "role": "default", "requires_train": True},
    # --- CPU-only helpers ---
    {"name": "convert_bbox", "module": "tools.bbox_utils", "attr": "convert_bbox",
     "hidden": set(), "role": "default", "requires_train": False},
    {"name": "validate_annotations", "module": "tools.bbox_utils", "attr": "validate_annotations",
     "hidden": set(), "role": "default", "requires_train": False},
    {"name": "compute_stats", "module": "tools.bbox_utils", "attr": "compute_stats",
     "hidden": set(), "role": "default", "requires_train": False},
]

_BY_NAME: dict[str, dict] = {e["name"]: e for e in _ENTRIES}
_CACHE: dict[str, ToolSpec] = {}


# ------------------------------------------------------------------
# Schema construction
# ------------------------------------------------------------------

def _schema_for(ann: Any) -> dict:
    """Map a (resolved) type annotation to a JSON-Schema fragment."""
    empty = inspect.Parameter.empty
    if ann is empty or ann is None or ann is type(None):
        return {}
    if isinstance(ann, str):
        return _schema_for_str(ann)

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    if origin is typing.Union or origin is getattr(types, "UnionType", ()):
        for a in args:
            if a is type(None):
                continue
            s = _schema_for(a)
            if s:
                return s
        return {}
    if origin in (list, tuple):
        item = _schema_for(args[0]) if args else {}
        return {"type": "array", "items": item or {}}
    if origin is dict:
        return {"type": "object"}

    if ann is bool:  # before int — bool is a subclass of int
        return {"type": "boolean"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is str:
        return {"type": "string"}
    if ann is dict:
        return {"type": "object"}
    if ann is list:
        return {"type": "array"}
    # PIL.Image.Image, pathlib.Path → a string path/URL over the wire
    if getattr(ann, "__name__", "") in ("Image", "Path"):
        return {"type": "string"}
    return {}


def _schema_for_str(ann: str) -> dict:
    """Best-effort mapping when annotations arrive as strings (eval failed)."""
    a = ann.replace(" ", "")
    if a.startswith(("list[", "List[", "tuple[")):
        return {"type": "array"}
    if a.startswith(("dict[", "Dict[")) or a == "dict":
        return {"type": "object"}
    if "bool" in a:
        return {"type": "boolean"}
    if "float" in a:
        return {"type": "number"}
    if "int" in a:
        return {"type": "integer"}
    if "str" in a or "Path" in a or "Image" in a:
        return {"type": "string"}
    return {}


def _param_docs(doc: str | None) -> dict[str, str]:
    """Parse a NumPy-style ``Parameters`` section → ``{param: description}``."""
    if not doc:
        return {}
    lines = doc.expandtabs().splitlines()
    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}:
            start = i + 2
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    cur: str | None = None
    base_indent: int | None = None
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        # A new section ("Returns" + dashes) at base indent ends the block.
        if (indent <= base_indent and i + 1 < len(lines)
                and set(lines[i + 1].strip()) == {"-"} and lines[i + 1].strip()):
            break
        m = re.match(r"^(\w+)\s*:", line.strip())
        if indent <= base_indent and m:
            cur = m.group(1)
            out[cur] = ""
        elif cur is not None:
            txt = line.strip()
            out[cur] = f"{out[cur]} {txt}".strip() if out[cur] else txt
        i += 1
    return out


def _short_desc(fn: Callable) -> str:
    """First paragraph of the function's docstring, whitespace-collapsed."""
    doc = inspect.getdoc(fn) or ""
    para: list[str] = []
    for line in doc.splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return " ".join(para)


def _build_schema(fn: Callable, hidden: frozenset) -> dict:
    try:
        sig = inspect.signature(fn, eval_str=True)
    except Exception:
        sig = inspect.signature(fn)
    docs = _param_docs(inspect.getdoc(fn))

    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if pname in hidden:
            continue
        schema = _schema_for(p.annotation)
        desc = docs.get(pname)
        if desc:
            schema = {**schema, "description": desc}
        props[pname] = schema
        if p.default is inspect.Parameter.empty:
            required.append(pname)

    out: dict = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def _materialize(entry: dict) -> ToolSpec:
    name = entry["name"]
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    mod = importlib.import_module(entry["module"])
    fn = getattr(mod, entry["attr"])
    hidden = frozenset(entry["hidden"])
    spec = ToolSpec(
        name=name,
        description=_short_desc(fn),
        parameters=_build_schema(fn, hidden),
        fn=fn,
        hidden=hidden,
        role=entry["role"],
        requires_train=entry["requires_train"],
    )
    _CACHE[name] = spec
    return spec


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def list_tools(include_train: bool = True) -> list[str]:
    """Names of registered tools (no import / materialisation)."""
    return [e["name"] for e in _ENTRIES if include_train or not e["requires_train"]]


def get_tools(include_train: bool = False) -> list[ToolSpec]:
    """Materialise and return tool specs.

    By default the torch-heavy tools (``train`` and the local-GPU model
    tools) are skipped, so this never imports ``torch`` on a CPU-only
    install.  Pass ``include_train=True`` to get the full set.
    """
    return [
        _materialize(e)
        for e in _ENTRIES
        if include_train or not e["requires_train"]
    ]


def get_tool(name: str) -> ToolSpec:
    """Materialise and return a single tool spec by name."""
    entry = _BY_NAME.get(name)
    if entry is None:
        raise KeyError(
            f"Unknown tool {name!r}. Available: {', '.join(list_tools())}"
        )
    return _materialize(entry)


def as_json_schema(include_train: bool = False) -> list[dict]:
    """List of ``{name, description, parameters}`` JSON-Schema specs."""
    return [t.as_dict() for t in get_tools(include_train=include_train)]


def call(name: str, **kwargs: Any) -> Any:
    """Invoke tool *name*, injecting hidden worker config for its role.

    Any hidden credential/endpoint param the caller did not pass is filled
    from :func:`tools.config.current` for the tool's role.  Pass it
    explicitly to override.  Unknown keyword arguments raise ``TypeError``.
    """
    spec = get_tool(name)
    sig_params = inspect.signature(spec.fn).parameters
    accepts_var_kw = any(
        p.kind is p.VAR_KEYWORD for p in sig_params.values()
    )
    if not accepts_var_kw:
        unknown = set(kwargs) - set(sig_params)
        if unknown:
            raise TypeError(
                f"{name}() got unexpected argument(s): "
                f"{', '.join(sorted(unknown))}"
            )

    if spec.hidden:
        cfg = current(spec.role)
        for field in config_field_names():  # backend, model_id, base_url, api_key
            if field in spec.hidden and field in sig_params and field not in kwargs:
                value = getattr(cfg, field)
                if value is not None:
                    kwargs[field] = value

    return spec.fn(**kwargs)
