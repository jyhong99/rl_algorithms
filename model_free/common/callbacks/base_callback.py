from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# =============================================================================
# Callback base
# =============================================================================
class BaseCallback:
    """
    Base callback interface for RL training loops.

    This class defines optional hook points that a Trainer can invoke at key
    moments (start/end, per-env-step, per-update, evaluation, checkpointing).

    Design
    ------
    - Hooks are *side-effect* oriented (logging, eval, saving, early stop, etc.).
    - Each hook returns a boolean:
        True  -> continue training
        False -> request early stop (Trainer should stop gracefully)

    Notes
    -----
    - All hooks are no-ops by default and return True.
    - The `trainer` object is duck-typed; callbacks should not assume a concrete
      Trainer class, only required attributes/methods that they access.

    Expected Trainer Contract (duck-typed)
    --------------------------------------
    Callbacks may access some subset of:
      - global_env_step: int
      - global_update_step: int
      - run_evaluation() -> Dict[str, Any]
      - save_checkpoint(path: Optional[str] = None) -> Optional[str]
      - logger.log(metrics: Dict[str, Any], step: int, prefix: str = "")
      - train_env / eval_env (optional)
      - algo (optional)

    Implementation guidance
    -----------------------
    - Keep hooks fast; heavy evaluation/checkpointing should be throttled by
      dedicated callback logic (e.g., every N steps).
    - Avoid raising exceptions for normal early-stop conditions; return False.
    """

    def on_train_start(self, trainer: Any) -> bool:
        """Called once before training begins."""
        return True

    def on_step(self, trainer: Any, transition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after each environment step.

        Parameters
        ----------
        trainer : Any
            Trainer instance (duck-typed).
        transition : Optional[Dict[str, Any]]
            Optional transition payload from the training loop. Typical keys:
            observations, actions, rewards, dones, infos, etc.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_update(self, trainer: Any, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called after each policy update (e.g., after PPO epochs, or one gradient update batch).

        Parameters
        ----------
        trainer : Any
            Trainer instance (duck-typed).
        metrics : Optional[Dict[str, Any]]
            Training metrics produced by the algorithm/core update.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        """
        Called after an evaluation run finishes.

        Parameters
        ----------
        trainer : Any
            Trainer instance (duck-typed).
        metrics : Dict[str, Any]
            Evaluation metrics (e.g., episodic return, length, success rate).

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
        trainer : Any
            Trainer instance (duck-typed).
        path : str
            Path where the checkpoint was written.

        Returns
        -------
        bool
            True to continue, False to request early stop.
        """
        return True

    def on_train_end(self, trainer: Any) -> bool:
        """Called once after training ends (whether normally or early-stopped)."""
        return True


# =============================================================================
# Callback composition
# =============================================================================
class CallbackList(BaseCallback):
    """
    Compose and dispatch multiple callbacks in order (short-circuit).

    If any callback hook returns False, dispatch stops immediately and the hook
    returns False. The Trainer should treat False as a request to stop training.

    Parameters
    ----------
    callbacks : Sequence[BaseCallback]
        Callbacks to dispatch. None entries are ignored.

    Notes
    -----
    - Ordering matters: callbacks are invoked in the given order.
    - This class does not swallow exceptions by default; if you want fault
      tolerance, wrap individual callbacks or implement a SafeCallback wrapper.
    """

    def __init__(self, callbacks: Sequence[Optional[BaseCallback]]):
        self.callbacks: List[BaseCallback] = [cb for cb in callbacks if cb is not None]

        # Defensive type check (helps catch accidental passing of non-callbacks)
        for i, cb in enumerate(self.callbacks):
            if not isinstance(cb, BaseCallback):
                raise TypeError(
                    f"callbacks[{i}] must be a BaseCallback, got: {type(cb).__name__}"
                )

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