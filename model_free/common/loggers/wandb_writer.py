from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .base_writer import Writer
from ..utils.log_utils import get_step

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class WandBWriter(Writer):
    """
    Weights & Biases backend writer.

    Notes
    -----
    - This writer is intended to be used either directly or wrapped by SafeWriter.
    - By default, it logs the row as-is (including meta keys) and uses `step` for W&B step.
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

        init_kwargs: Dict[str, Any] = dict(
            project=str(project),
            entity=entity,
            group=group,
            tags=list(tags) if tags is not None else None,
            name=name,
            dir=str(run_dir),
        )
        if mode is not None:
            init_kwargs["mode"] = mode
        if resume is not None:
            init_kwargs["resume"] = resume

        # remove None values for wandb.init compatibility
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        wandb.init(**init_kwargs)
        self._enabled = True

    def write(self, row: Dict[str, float]) -> None:
        if not self._enabled or wandb is None:
            return
        step = get_step(row)
        wandb.log(dict(row), step=step)

    def flush(self) -> None:
        # wandb flushes internally; no-op
        return

    def close(self) -> None:
        if self._enabled and wandb is not None:
            wandb.finish()
        self._enabled = False
