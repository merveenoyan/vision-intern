"""Shared pytest configuration for tests/.

Adds the project root to sys.path so package-style imports
(``workflows.vlm_judge``, ``tools.dataset_utils``) resolve without
installing the package, and stubs GPU-heavy libraries so tests that
exercise pure-Python logic work on a CPU-only machine.
"""
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _GpuStub(types.ModuleType):
    """Recursive module stub that satisfies importlib/isinstance/issubclass.

    Three constraints must be met simultaneously:

    1. ``importlib.util.find_spec`` (Python 3.11+) raises ``ValueError`` when
       ``module.__spec__`` is not a proper ``ModuleSpec``.

    2. ``isinstance(x, torch.Tensor)`` raises ``TypeError`` when
       ``torch.Tensor`` is a MagicMock (not a type).

    3. ``issubclass(t, torch.nn.Module)`` raises ``TypeError`` for the same
       reason — and ``torch.nn`` must be a sub-stub, not a flat MagicMock,
       so that ``torch.nn.Module`` can return a real type.

    Solution: CamelCase attribute access returns an empty type (class); all
    other attribute access returns a nested _GpuStub so chained access like
    ``torch.nn.Module`` works recursively.  Both are idempotent (cached via
    ``object.__setattr__``).
    """

    def __getattr__(self, name: str) -> object:
        if name[:1].isupper():
            # Class-like name → real empty type for isinstance/issubclass.
            # Nothing will ever be an instance of it, which is correct on a
            # CPU-only machine.
            val: object = type(name, (), {"__module__": self.__name__})
        else:
            # Module-like name → nested stub so torch.nn.Module chains work.
            val = _make_module_stub(f"{self.__name__}.{name}")
        object.__setattr__(self, name, val)
        return val


def _make_module_stub(name: str) -> _GpuStub:
    stub = _GpuStub(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    stub.__path__ = []
    return stub


# Pre-stub GPU libraries before any module-level `import torch` runs.
# Only applied when the real library is absent so real-GPU CI keeps the
# actual packages.
for _mod in ("torch", "torchvision", "accelerate"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _make_module_stub(_mod)
