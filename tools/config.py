"""Worker configuration for the agent tool layer.

The VLM-backed tools and workflows take ``backend`` / ``model_id`` /
``base_url`` / ``api_key`` on every call.  Those are *infrastructure*
choices — which endpoint serves the labeller, which the judges — not
something an orchestrating agent should fill in per call (and an API key
must never end up in a tool's JSON schema).

This module holds that config out of band.  :func:`tools.registry.call`
injects it for any hidden credential param the caller didn't override, so
the agent-facing schema stays down to the *task* arguments.

Roles
-----
``"default"``   VLM tools (``vlm_detect``, ``ocr_judge``).
``"labeller"``  the ``label_dataset`` workflow (the larger worker).
``"judge"``     the ``judge_labels`` workflow (the smaller verifier).

Configure once at start-up::

    from tools import configure, ToolConfig

    configure(
        labeller=ToolConfig(base_url="https://router.huggingface.co/v1",
                            model_id="Qwen/Qwen3-VL-8B-Instruct"),
        judge=ToolConfig(base_url="http://localhost:8084/v1",
                        model_id="Qwen3-VL-4B-Instruct-Q8_0.gguf"),
    )

Any field left ``None`` falls through to the underlying function's own
default (``model_id``), the HF Inference Providers URL (``base_url``), or
the ``HF_TOKEN`` / ``OPENAI_API_KEY`` env fallback already in
:func:`tools.vlm_client._resolve_api_key` (``api_key``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

ROLES = ("default", "labeller", "judge")


@dataclass(frozen=True)
class ToolConfig:
    """Where a VLM worker runs.  All fields optional; ``None`` means *fall
    through to the underlying default / env*."""

    backend: str | None = "openai"
    model_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    def merge(self, **overrides: object) -> "ToolConfig":
        """Return a copy with non-``None`` *overrides* applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


def _from_env() -> ToolConfig:
    return ToolConfig(
        backend=os.environ.get("VISION_AGENT_BACKEND", "openai"),
        model_id=os.environ.get("VISION_AGENT_MODEL"),
        base_url=os.environ.get("VISION_AGENT_BASE_URL"),
        # api_key intentionally left None — vlm_client resolves HF_TOKEN /
        # OPENAI_API_KEY at call time so the token never sits in this object.
        api_key=None,
    )


# Per-role config; every role starts from the environment defaults.
_CONFIG: dict[str, ToolConfig] = {role: _from_env() for role in ROLES}


def configure(
    default: ToolConfig | None = None,
    labeller: ToolConfig | None = None,
    judge: ToolConfig | None = None,
) -> None:
    """Set the worker config for one or more roles.

    Unspecified roles are left unchanged.  ``labeller`` / ``judge`` fall
    back to ``default`` for any field they leave ``None`` only if you pass
    a populated ``default`` — otherwise each role is independent.
    """
    if default is not None:
        _CONFIG["default"] = default
    if labeller is not None:
        _CONFIG["labeller"] = labeller
    if judge is not None:
        _CONFIG["judge"] = judge


def current(role: str = "default") -> ToolConfig:
    """Return the active :class:`ToolConfig` for *role*."""
    if role not in _CONFIG:
        role = "default"
    return _CONFIG[role]


def reset() -> None:
    """Restore every role to the environment defaults (used in tests)."""
    for role in ROLES:
        _CONFIG[role] = _from_env()


def config_field_names() -> tuple[str, ...]:
    """Names of the credential/endpoint fields a :class:`ToolConfig` can fill."""
    return tuple(f.name for f in fields(ToolConfig))
