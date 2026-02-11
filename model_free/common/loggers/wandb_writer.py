from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .base_writer import Writer
from ..utils.logger_utils import _get_step

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore[assignment]


class WandBWriter(Writer):
    """
    Weights & Biases (W&B) writer backend for scalar metric logging.

    This writer forwards each metric row to W&B via :func:`wandb.log`. The row is
    logged largely "as-is" (including meta fields like ``step``, ``wall_time``,
    and ``timestamp``). The W&B step is taken from `_get_step(row)` to remain
    consistent with other sinks.

    Parameters
    ----------
    run_dir : str
        Local directory used by W&B for storing run artifacts and metadata
        (passed to :func:`wandb.init` via the ``dir`` argument).
    project : str
        W&B project name. This is required.
    entity : str, optional
        W&B entity (user or team) under which the run should be logged.
    group : str, optional
        W&B group name for grouping related runs (e.g., sweeps or ablations).
    tags : Sequence[str], optional
        Optional list/sequence of string tags applied to the run.
    name : str, optional
        Optional human-readable run name in the W&B UI.
    mode : str, optional
        W&B mode passed through to :func:`wandb.init` (e.g., "online", "offline", "disabled").
        Exact allowed values depend on the installed W&B version.
    resume : str, optional
        Resume behavior passed through to :func:`wandb.init` (e.g., "allow", "must", "never").
        Exact semantics depend on W&B.

    Attributes
    ----------
    _enabled : bool
        Internal flag indicating whether logging is active. Set False after `close()`.

    Raises
    ------
    RuntimeError
        If the `wandb` package is not available.

    Notes
    -----
    - This writer is typically safe to wrap in `SafeWriter` if you want hard isolation
      from transient W&B/network failures. In strict pipelines, you may prefer to
      leave it unwrapped and let exceptions propagate.
    - This class does not implement an explicit `flush()`. W&B manages buffering
      internally; `flush()` is a no-op.
    - By default, meta keys are logged too. If you prefer to drop meta keys, do it
      at the `Logger` level or add a `_split_meta` call here.
    """

    def __init__(
        self,
        *,
        run_dir: str,
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        mode: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> None:
        if wandb is None:
            raise RuntimeError("wandb is not available. Install with `pip install wandb`.")

        init_kwargs: Dict[str, Any] = {
            "project": str(project),
            "entity": entity,
            "group": group,
            "tags": list(tags) if tags is not None else None,
            "name": name,
            "dir": str(run_dir),
        }
        if mode is not None:
            init_kwargs["mode"] = mode
        if resume is not None:
            init_kwargs["resume"] = resume

        # Remove None values for `wandb.init` compatibility.
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        wandb.init(**init_kwargs)
        self._enabled = True

    def write(self, row: Dict[str, float]) -> None:
        """
        Log one metric row to W&B.

        Parameters
        ----------
        row : Dict[str, float]
            Flat mapping of keys to scalar floats. May include meta keys.
            The W&B step is derived from this row using `_get_step(row)`.

        Notes
        -----
        - If this writer has been closed (`_enabled=False`), this is a no-op.
        - The input is defensively copied via `dict(row)` before sending to W&B.
        """
        if not self._enabled or wandb is None:
            return
        step = int(_get_step(row))
        wandb.log(dict(row), step=step)

    def flush(self) -> None:
        """
        Flush pending logs (no-op).

        Notes
        -----
        W&B manages buffering and synchronization internally, so this method
        intentionally does nothing.
        """
        return

    def close(self) -> None:
        """
        Finish the W&B run (best-effort) and disable further logging.

        Notes
        -----
        - After `close()`, `write()` becomes a no-op.
        - Exceptions from `wandb.finish()` are not caught here by design; if you want
          failure isolation, wrap this writer with `SafeWriter` or catch in `Logger`.
        """
        if self._enabled and wandb is not None:
            wandb.finish()
        self._enabled = False
