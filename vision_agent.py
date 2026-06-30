"""vision-agent — top-level agent entry point.

A convenience shim so an orchestrating agent can use the toolkit without
knowing the internal ``tools`` / ``workflows`` layout::

    import vision_agent as va

    va.configure(labeller=va.ToolConfig(model_id="Qwen/Qwen3-VL-8B-Instruct"))
    tools = va.get_tools()                 # JSON-schema'd, torch-free by default
    boxes = va.call("vlm_detect", image="photo.jpg", classes=["cat", "dog"])

Everything here is re-exported from :mod:`tools.registry` and
:mod:`tools.config`; nothing imports ``torch`` until a training tool is
actually invoked.
"""

from __future__ import annotations

from tools.config import ToolConfig, configure, current
from tools.registry import (
    ToolSpec,
    as_json_schema,
    call,
    get_tool,
    get_tools,
    list_tools,
)

__all__ = [
    "ToolConfig",
    "ToolSpec",
    "as_json_schema",
    "call",
    "configure",
    "current",
    "get_tool",
    "get_tools",
    "list_tools",
]
