"""
common package
==============

This package was extracted/repurposed from a larger codebase where imports often
used the namespace `model_free.common.*`.

To keep backward compatibility **without touching non-__init__.py files**, we
install a lightweight alias so that:

    import model_free.common.utils
    from model_free.common.optimizers import build_optimizer
    ...

continue to work when this package is installed/imported as `common`.

Implementation details
----------------------
- We register `model_free` and `model_free.common` in `sys.modules`.
- `model_free.common` is pointed at this `common` package module object.
- Submodules will resolve naturally once imported under `common.*`.

This is intentionally small and defensive.
"""

from __future__ import annotations

import sys
import types


# -----------------------------------------------------------------------------
# Backward-compatible namespace alias: model_free.common -> common
# -----------------------------------------------------------------------------
def _install_model_free_alias() -> None:
    """
    Install sys.modules aliases so legacy absolute imports under `model_free.common`
    resolve to this package.

    This does NOT import all submodules eagerly; it simply ensures Python's import
    machinery can find the top-level packages.
    """
    # If user already has a real `model_free` package installed, do not clobber it.
    if "model_free" in sys.modules and not isinstance(sys.modules["model_free"], types.ModuleType):
        return

    # Ensure `model_free` exists as a module container.
    if "model_free" not in sys.modules:
        sys.modules["model_free"] = types.ModuleType("model_free")

    # Point `model_free.common` to this `common` package.
    # `__name__` is "common" here; `sys.modules[__name__]` is the actual module object.
    sys.modules["model_free.common"] = sys.modules[__name__]

    # Also expose as attribute: model_free.common
    try:
        sys.modules["model_free"].common = sys.modules[__name__]  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; failing to set attribute is not fatal if sys.modules mapping works.
        pass


_install_model_free_alias()


# -----------------------------------------------------------------------------
# Public re-exports (keep minimal; avoid heavy imports at package import time)
# -----------------------------------------------------------------------------
__all__ = [
    "buffers",
    "callbacks",
    "loggers",
    "networks",
    "noises",
    "optimizers",
    "policies",
    "trainers",
    "wrappers",
]
