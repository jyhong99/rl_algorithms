from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Dict, List, Optional

from .base_callback import BaseCallback
from ..utils.callback_utils import IntervalGate, _safe_env_step


class CheckpointCallback(BaseCallback):
    """
    Periodically save checkpoints on an environment-step schedule, with optional rotation.

    This callback triggers checkpoint saves based on **environment steps** using an
    :class:`~IntervalGate`. When a checkpoint is successfully saved, the path is
    recorded and older checkpoints can be deleted to keep only the most recent
    ``keep_last`` checkpoints.

    Parameters
    ----------
    save_every:
        Save interval in **environment steps**. If ``save_every <= 0``,
        checkpointing is disabled (callback becomes a no-op).
    keep_last:
        Number of most recent checkpoint paths to keep. If ``keep_last <= 0``,
        rotation (deletion of old checkpoints) is disabled.

    Attributes
    ----------
    save_every:
        Configured environment-step interval for saving checkpoints.
    keep_last:
        Configured number of checkpoints to keep.
    _gate:
        Interval gate controlling when to fire checkpoint saves.
    _paths:
        FIFO list of checkpoint paths for rotation.

    Trainer contract (duck-typed)
    -----------------------------
    The trainer is expected to provide a callable ``save_checkpoint`` method:

    - Preferred: ``trainer.save_checkpoint() -> Optional[str]``
    - Fallback:  ``trainer.save_checkpoint(path=None) -> Optional[str]``

    Optional trainer fields used on train start (best-effort scan):
    - ``trainer.ckpt_dir``: directory containing checkpoint files
    - ``trainer.checkpoint_prefix``: filename prefix used for checkpoint files (default: "ckpt")

    Logging
    -------
    On each successful checkpoint save, emits a minimal scalar metric:

    - ``sys/checkpoint/saved = 1.0``

    Notes
    -----
    - This callback does **not** invoke ``trainer.callbacks.on_checkpoint(...)`` to
      avoid recursion / double dispatch. If you need checkpoint events, the trainer
      should broadcast them explicitly.
    - Deletion of old checkpoints is best-effort and never raises.
    """

    def __init__(self, save_every: int = 100_000, keep_last: int = 5) -> None:
        self.save_every = int(save_every)
        self.keep_last = int(keep_last)

        # Gate policy:
        #   mode="delta" => fires when (step - last) >= every, then advances last.
        self._gate = IntervalGate(every=self.save_every, mode="delta")

        # FIFO of checkpoint paths used for rotation.
        self._paths: List[str] = []

    # =========================================================================
    # Internal filesystem helpers
    # =========================================================================
    @staticmethod
    def _best_effort_delete(path: str) -> bool:
        """
        Best-effort delete for a file or directory (never raises).

        Parameters
        ----------
        path:
            Target path to delete.

        Returns
        -------
        bool
            True if deletion was attempted or the target was safely absent.
            False if the input path is empty or an exception occurred.
        """
        try:
            if not path:
                return False

            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                return True

            if os.path.exists(path):
                os.remove(path)
                return True

            return False
        except Exception:
            return False

    @staticmethod
    def _best_effort_delete_checkpoint_family(path: str) -> None:
        """
        Best-effort delete a checkpoint and associated sibling artifacts.

        This function deletes:
        - The main checkpoint file itself, and
        - Any sibling artifacts that share the same "stem", such as:

          - ``ckpt_000000001500.pt``
          - ``ckpt_000000001500.json``
          - ``ckpt_000000001500_algo.pt``
          - ``ckpt_000000001500_*``

        Parameters
        ----------
        path:
            Path to the main checkpoint artifact (file or directory).

        Notes
        -----
        - If ``path`` is a directory, the entire directory is removed.
        - If ``path`` is a file, we remove it and also glob-delete siblings
          with patterns ``stem + ".*"`` and ``stem + "_*"``.
        - All errors are swallowed (best-effort).
        """
        try:
            if not path:
                return

            # Directory checkpoint (some trainers save as a folder).
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                return

            # File checkpoint: delete the file and sibling artifacts.
            if os.path.isfile(path):
                stem, _ext = os.path.splitext(path)

                # Delete the main file first.
                try:
                    os.remove(path)
                except Exception:
                    pass

                # Delete siblings with the same stem.
                patterns = [
                    stem + ".*",  # e.g., ckpt_xxxx.json, ckpt_xxxx.npz, ...
                    stem + "_*",  # e.g., ckpt_xxxx_algo.pt, ckpt_xxxx_rms.npz, ...
                ]
                for pat in patterns:
                    for p in glob.glob(pat):
                        try:
                            if os.path.isdir(p):
                                shutil.rmtree(p, ignore_errors=True)
                            else:
                                os.remove(p)
                        except Exception:
                            pass
                return

            # If it doesn't exist, nothing to do.
            return
        except Exception:
            return

    def _rotate_checkpoints(self) -> None:
        """
        Enforce the ``keep_last`` retention policy.

        Notes
        -----
        - Rotation is disabled if ``keep_last <= 0``.
        - Oldest paths are removed first (FIFO).
        - Deletion is best-effort and never raises.
        """
        if self.keep_last <= 0:
            return

        while len(self._paths) > self.keep_last:
            old = self._paths.pop(0)
            if old:
                self._best_effort_delete_checkpoint_family(old)

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================
    def on_train_start(self, trainer: Any) -> bool:
        """
        Initialize the interval gate and optionally sync existing checkpoints.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). May provide ``ckpt_dir`` and
            ``checkpoint_prefix`` for best-effort checkpoint scanning.

        Returns
        -------
        bool
            Always True (this callback does not request early stop).

        Notes
        -----
        - The interval gate ``last`` is aligned to the largest multiple of
          ``save_every`` not exceeding the current env step to avoid immediate
          firing if resuming from a checkpoint.
        - If ``trainer.ckpt_dir`` exists, we scan for existing checkpoint files
          matching the prefix and populate ``self._paths`` for retention rotation.
        """
        # If disabled, keep gate state consistent and exit.
        if self.save_every <= 0:
            self._gate.every = self.save_every
            self._gate.last = 0
            return True

        step = _safe_env_step(trainer)
        if step < 0:
            step = 0

        self._gate.every = self.save_every
        self._gate.last = (step // self.save_every) * self.save_every

        # Best-effort: scan existing checkpoints to sync rotation state.
        try:
            ckpt_dir = getattr(trainer, "ckpt_dir", None)
            prefix = getattr(trainer, "checkpoint_prefix", "ckpt")
            if isinstance(ckpt_dir, str) and ckpt_dir and os.path.isdir(ckpt_dir):
                paths = glob.glob(os.path.join(ckpt_dir, f"{prefix}*"))
                paths = [p for p in paths if os.path.isfile(p)]
                # Oldest first (mtime).
                paths.sort(key=os.path.getmtime)
                self._paths = paths
                self._rotate_checkpoints()
        except Exception:
            pass

        return True

    # =========================================================================
    # Internal checkpoint helpers
    # =========================================================================
    def _save_checkpoint(self, trainer: Any) -> Optional[str]:
        """
        Best-effort checkpoint save.

        Supports trainer implementations exposing either:
        - ``save_checkpoint()`` or
        - ``save_checkpoint(path=None)``

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).

        Returns
        -------
        Optional[str]
            Non-empty checkpoint path string on success; otherwise None.

        Notes
        -----
        Any exceptions are swallowed (best-effort).
        """
        save_fn = getattr(trainer, "save_checkpoint", None)
        if not callable(save_fn):
            return None

        try:
            path = save_fn()
        except TypeError:
            try:
                path = save_fn(path=None)
            except Exception:
                return None
        except Exception:
            return None

        return path if isinstance(path, str) and path else None

    # =========================================================================
    # Step hook
    # =========================================================================
    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a checkpoint when the interval gate fires.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed). Must provide ``save_checkpoint`` to save.
        transition:
            Unused transition payload (accepted for hook compatibility).

        Returns
        -------
        bool
            Always True (this callback does not request early stop).

        Notes
        -----
        - Uses :func:`safe_env_step` to retrieve the current step robustly.
        - Logs only a minimal scalar (no path strings) to avoid log spam.
        """
        if self.save_every <= 0:
            return True

        step = _safe_env_step(trainer)
        if step <= 0:
            return True

        if not self._gate.ready(step):
            return True

        path = self._save_checkpoint(trainer)
        if path is not None:
            self._paths.append(path)
            self._rotate_checkpoints()
            self.log(trainer, {"checkpoint/saved": 1.0}, step=step, prefix="sys/")

        return True
