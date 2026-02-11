from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# =============================================================================
# Callback base
# =============================================================================
class BaseCallback:
    """
    Base callback interface for RL training loops.

    Callbacks provide hook points that a training loop (``trainer``) can invoke
    at key lifecycle events. Hooks are intended for side effects such as logging,
    evaluation, checkpointing, scheduling, and early stopping.

    Design
    ------
    - Hooks are *optional* and are no-ops by default.
    - Each hook returns a boolean control signal:

      - ``True``  : continue training
      - ``False`` : request an early stop (trainer should stop gracefully)

    - The ``trainer`` argument is intentionally **duck-typed**: callbacks should
      only rely on the attributes/methods they actually access.

    Expected Trainer Contract (duck-typed)
    --------------------------------------
    A trainer that uses callbacks typically provides some subset of:

    - ``global_env_step: int``:
        Monotonic count of environment steps across all episodes.
    - ``global_update_step: int``:
        Monotonic count of parameter updates / optimization steps.
    - ``run_evaluation() -> Dict[str, Any]``:
        Executes evaluation and returns metrics.
    - ``save_checkpoint(path: Optional[str] = None) -> Optional[str]``:
        Saves model/state and returns the checkpoint path (or None).
    - ``logger`` with method:
        ``logger.log(metrics: Dict[str, Any], step: int, prefix: str = "")``.
    - Optional fields such as ``train_env``, ``eval_env``, ``algo``, etc.

    Performance guidance
    --------------------
    - Keep hooks lightweight. Heavy operations (evaluation, I/O) should be
      throttled by the callback (e.g., "every N steps/updates").
    - Use ``False`` returns for normal early-stop conditions rather than raising.

    Notes
    -----
    - This base class does not enforce an abstract interface to keep integration
      friction low.
    - Individual callbacks may choose to be strict about required trainer fields.
    """

    def on_train_start(self, trainer: Any) -> bool:
        """
        Called once before training begins.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after each environment step.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).
        transition:
            Optional transition payload emitted by the training loop.
            Typical keys (convention-dependent): ``observations``, ``actions``,
            ``rewards``, ``dones``, ``infos``, etc.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after each policy/parameter update.

        Examples include:
        - After a PPO update phase (multiple epochs/minibatches)
        - After one gradient step for off-policy algorithms

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).
        metrics:
            Optional training metrics produced by the update step.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Called after an evaluation run completes.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).
        metrics:
            Evaluation metrics (e.g., episodic return/length, success rate,
            constraint violations, etc.).

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_checkpoint(self, trainer: Any, path: str) -> bool:
        """
        Called after a checkpoint is saved.

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).
        path:
            Path where the checkpoint was written.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_train_end(self, trainer: Any) -> bool:
        """
        Called once after training ends (normal completion or early stop).

        Parameters
        ----------
        trainer:
            Trainer object (duck-typed).

        Returns
        -------
        bool
            True to indicate successful teardown, False to request early stop
            (typically ignored at this stage, but kept for symmetry).
        """
        return True

    def log(
        self,
        trainer: Any,
        metrics: Dict[str, Any],
        *,
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Best-effort logger dispatch.

        This helper attempts to call ``trainer.logger.log(...)`` if present.
        It is intentionally fault-tolerant: missing logger or logging failures
        will not raise.

        Expected logger interface (duck-typed)
        --------------------------------------
        ``trainer.logger.log(metrics: dict, step: int, prefix: str = "")``

        Parameters
        ----------
        trainer:
            Trainer object holding a ``.logger`` attribute.
        metrics:
            Metrics payload (typically scalars and small JSON-friendly objects).
        step:
            Global step associated with the metrics (e.g., env_step or update_step).
        prefix:
            Optional namespace prefix (e.g., ``"train/"``, ``"eval/"``, ``"sys/"``).

        Notes
        -----
        - Exceptions are swallowed by design: callbacks should not crash training.
        - If the logger is missing or does not implement ``.log``, this is a no-op.
        """
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return

        fn = getattr(logger, "log", None)
        if not callable(fn):
            return

        try:
            fn(metrics, step=step, prefix=prefix)
        except Exception:
            # Callbacks must not crash the training loop.
            return


# =============================================================================
# Callback composition
# =============================================================================
class CallbackList(BaseCallback):
    """
    Compose and dispatch multiple callbacks in order (short-circuit).

    Each hook is forwarded to the contained callbacks in the given order.
    If any callback returns ``False``, dispatch stops immediately and the hook
    returns ``False`` to the trainer.

    Parameters
    ----------
    callbacks:
        Sequence of callbacks to dispatch. ``None`` entries are ignored.

    Notes
    -----
    - Ordering matters: callbacks are invoked in the provided order.
    - This class does **not** swallow exceptions raised by contained callbacks.
      If you need fault tolerance, wrap callbacks or implement a safe wrapper.
    """

    def __init__(self, callbacks: Sequence[Optional[BaseCallback]]) -> None:
        self.callbacks: List[BaseCallback] = [cb for cb in callbacks if cb is not None]

        # Defensive type check: catch accidental passing of non-callback objects early.
        for i, cb in enumerate(self.callbacks):
            if not isinstance(cb, BaseCallback):
                raise TypeError(f"callbacks[{i}] must be a BaseCallback, got: {type(cb).__name__}")

    def on_train_start(self, trainer: Any) -> bool:
        for cb in self.callbacks:
            if not cb.on_train_start(trainer):
                return False
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        for cb in self.callbacks:
            if not cb.on_step(trainer, transition):
                return False
        return True

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        for cb in self.callbacks:
            if not cb.on_update(trainer, metrics):
                return False
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        for cb in self.callbacks:
            if not cb.on_eval_end(trainer, metrics):
                return False
        return True

    def on_checkpoint(self, trainer: Any, path: str) -> bool:
        for cb in self.callbacks:
            if not cb.on_checkpoint(trainer, path):
                return False
        return True

    def on_train_end(self, trainer: Any) -> bool:
        for cb in self.callbacks:
            if not cb.on_train_end(trainer):
                return False
        return True
