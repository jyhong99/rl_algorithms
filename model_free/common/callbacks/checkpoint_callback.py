from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import shutil
import glob

from .base_callback import BaseCallback
from ..utils.callback_utils import safe_int_attr, IntervalGate
from ..utils.log_utils import log


class CheckpointCallback(BaseCallback):
    """
    Periodically save checkpoints on an environment-step schedule, with optional rotation.

    Parameters
    ----------
    save_every : int, default=100_000
        Save interval in *environment steps*. If <= 0, checkpointing is disabled.
    keep_last : int, default=5
        Keep only the most recent N checkpoint paths. If <= 0, rotation is disabled.

    Trainer contract (duck-typed)
    -----------------------------
    trainer.save_checkpoint(...) -> Optional[str]
        - Preferred: save_checkpoint() returning a path string.
        - Fallback: save_checkpoint(path=None) if signature requires path.

    Logging
    -------
    Emits:
      sys/checkpoint/saved = 1.0

    Notes
    -----
    This callback does NOT call trainer.callbacks.on_checkpoint(...) to avoid recursion/double-dispatch.
    If you need checkpoint events, the Trainer should broadcast them.
    """

    def __init__(self, save_every: int = 100_000, keep_last: int = 5) -> None:
        self.save_every = int(save_every)
        self.keep_last = int(keep_last)

        # Gate controls schedule triggering.
        # mode="delta": triggers when (step - last) >= every; then advances last internally.
        self._gate = IntervalGate(every=self.save_every, mode="delta")

        # FIFO of checkpoint paths for rotation.
        self._paths: List[str] = []

    # =========================================================================
    # Internal filesystem helper (merged from standalone function)
    # =========================================================================
    @staticmethod
    def _best_effort_delete(path: str) -> bool:
        """
        Best-effort delete for a file or directory. Never raises.

        Returns True if deletion was attempted (or target was absent in a safe way),
        False if input is empty or an exception occurred.
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

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def on_train_start(self, trainer: Any) -> bool:
        if self.save_every <= 0:
            self._gate.every = self.save_every
            self._gate.last = 0
            return True

        step = safe_int_attr(trainer)
        if step < 0:
            step = 0

        self._gate.every = self.save_every
        self._gate.last = (step // self.save_every) * self.save_every

        # -----------------------------
        # NEW: 기존 ckpt 스캔해서 _paths 동기화
        # -----------------------------
        try:
            ckpt_dir = getattr(trainer, "ckpt_dir", None)
            prefix = getattr(trainer, "checkpoint_prefix", "ckpt")
            if isinstance(ckpt_dir, str) and ckpt_dir and os.path.isdir(ckpt_dir):
                paths = glob.glob(os.path.join(ckpt_dir, f"{prefix}*"))
                paths = [p for p in paths if os.path.isfile(p)]
                paths.sort(key=os.path.getmtime)  # 오래된 순
                self._paths = paths
                self._rotate_checkpoints()
        except Exception:
            # best-effort
            pass

        return True

    # =========================================================================
    # Internal checkpoint helpers
    # =========================================================================
    def _save_checkpoint(self, trainer: Any) -> Optional[str]:
        """
        Best-effort checkpoint save supporting:
          - save_checkpoint()
          - save_checkpoint(path=None)

        Returns a non-empty path string on success, else None.
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

    @staticmethod
    def _best_effort_delete_checkpoint_family(path: str) -> None:
        """
        Delete checkpoint file and any sibling artifacts that share the same stem.
        Example:
          ckpt_000000001500.pt
          ckpt_000000001500.json
          ckpt_000000001500_algo.pt
          ckpt_000000001500_*   (etc.)
        """
        try:
            if not path:
                return

            # If it's a directory, nuke the directory.
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                return

            # If it's a file, delete file + siblings.
            if os.path.isfile(path):
                stem, _ext = os.path.splitext(path)

                # Delete the main file first
                try:
                    os.remove(path)
                except Exception:
                    pass

                # Delete siblings with same stem
                patterns = [
                    stem + ".*",     # ckpt_xxxx.json, ckpt_xxxx.npz, ...
                    stem + "_*",     # ckpt_xxxx_algo.pt, ckpt_xxxx_rms.npz, ...
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

            # If it doesn't exist, nothing to do
            return
        except Exception:
            return

    def _rotate_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return

        while len(self._paths) > self.keep_last:
            old = self._paths.pop(0)
            if old:
                self._best_effort_delete_checkpoint_family(old)

    # =========================================================================
    # Step hook
    # =========================================================================
    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a checkpoint when the interval gate fires.
        """
        if self.save_every <= 0:
            return True

        step = safe_int_attr(trainer)
        if step <= 0:
            return True

        if not self._gate.ready(step):
            return True

        path = self._save_checkpoint(trainer)
        if path is not None:
            self._paths.append(path)
            self._rotate_checkpoints()

            # Minimal metric: avoid spamming logs with path strings.
            log(trainer, {"checkpoint/saved": 1.0}, step=step, prefix="sys/")

        return True