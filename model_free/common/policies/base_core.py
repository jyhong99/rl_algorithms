from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence

import torch as th
import torch.nn as nn
from torch.optim import Optimizer

from model_free.common.optimizers.optimizer_builder import (
    build_optimizer,
    clip_grad_norm,
    load_optimizer_state_dict,
    optimizer_state_dict,
)
from model_free.common.optimizers.scheduler_builder import (
    build_scheduler,
    load_scheduler_state_dict,
    scheduler_state_dict,
)


class BaseCore(ABC):
    """
    Base class for update engines ("cores").

    A "core" is the unit responsible for performing parameter updates from data
    batches (e.g., PPO update step, SAC update step, DQN update step). This base
    class provides shared infrastructure that most RL training cores need:

    - A reference to `head` (duck-typed container of networks and utilities).
    - Normalized `device` selection.
    - Optional AMP (automatic mixed precision) support via GradScaler.
    - A monotonically increasing update-call counter.
    - AMP-safe gradient clipping helper.
    - Target-network update utilities (hard/soft updates + freezing).
    - Optimizer/scheduler checkpoint serialization helpers.

    Parameters
    ----------
    head : Any
        A duck-typed container typically holding neural networks (e.g., actor,
        critic) and optional helper methods. Commonly expected attributes:

        - device : torch.device or str, optional
            Device used for training. If missing, defaults to CPU.
        - freeze_target(module) : callable, optional
            Custom target-freezing logic (e.g., disabling grads, eval mode).
        - hard_update(target, source) : callable, optional
            Custom hard update logic for target networks.
        - soft_update(target, source, tau=...) : callable, optional
            Custom Polyak averaging logic.

    use_amp : bool, default=False
        If True and CUDA is available, enables AMP with a GradScaler. AMP is
        automatically disabled on CPU or when CUDA is not available.

    Notes
    -----
    - Concrete subclasses must implement `update_from_batch`.
    - The update counter `_update_calls` is commonly used to schedule target
      network updates (e.g., every N gradient steps).
    """

    def __init__(self, *, head: Any, use_amp: bool = False) -> None:
        self.head = head

        # -------------------------
        # Device normalization
        # -------------------------
        dev = getattr(head, "device", th.device("cpu"))
        self.device = dev if isinstance(dev, th.device) else th.device(str(dev))

        # AMP is only meaningful on CUDA
        self.use_amp = bool(use_amp) and (self.device.type == "cuda") and th.cuda.is_available()

        # -------------------------
        # GradScaler (safe across torch variants)
        # -------------------------
        try:
            self.scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)
        except Exception:
            # Fallback for some torch builds/versions
            self.scaler = th.amp.GradScaler(enabled=self.use_amp)  # type: ignore[attr-defined]

        self._update_calls: int = 0

    # ---------------------------------------------------------------------
    # Bookkeeping
    # ---------------------------------------------------------------------
    @property
    def update_calls(self) -> int:
        """
        Number of times the core performed an update step.

        Returns
        -------
        update_calls : int
            Count of completed update calls, typically incremented by subclasses
            after a successful parameter update.
        """
        return int(self._update_calls)

    def _bump(self) -> None:
        """
        Increment internal update counter.

        Notes
        -----
        Subclasses typically call this once per "update step" (e.g., after
        optimizer.step()) so that target-network update schedules remain correct.
        """
        self._update_calls += 1

    # ---------------------------------------------------------------------
    # Gradient clipping
    # ---------------------------------------------------------------------
    def _clip_params(
        self,
        params: Any,
        *,
        max_grad_norm: float,
        optimizer: Optional[Any] = None,
    ) -> None:
        """
        Clip gradients in-place (optionally AMP-safe).

        Parameters
        ----------
        params : Any
            Iterable of parameters whose gradients will be clipped
            (e.g., `module.parameters()`).

        max_grad_norm : float
            Maximum allowed norm. If `max_grad_norm <= 0`, clipping is disabled.

        optimizer : Optional[Any], default=None
            Optimizer instance. If AMP is enabled, it is passed to the clipping
            utility so that gradients can be unscaled before clipping.

        Notes
        -----
        - If AMP is enabled, gradients are unscaled (via GradScaler) before clipping.
        - This method delegates to your project utility `clip_grad_norm`, which
          is expected to accept `(scaler=..., optimizer=...)`.
        """
        mg = float(max_grad_norm)
        if mg <= 0.0:
            return

        if self.use_amp:
            clip_grad_norm(params, mg, scaler=self.scaler, optimizer=optimizer)
        else:
            clip_grad_norm(params, mg)

    # ---------------------------------------------------------------------
    # Target network freezing
    # ---------------------------------------------------------------------
    @staticmethod
    def _freeze_module_fallback(module: nn.Module) -> None:
        """
        Fallback target freezing.

        This fallback:
        - disables gradients (`requires_grad=False`) for all parameters
        - switches the module into eval mode

        Parameters
        ----------
        module : nn.Module
            Target module to freeze.
        """
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    def _freeze_target(self, module: nn.Module) -> None:
        """
        Freeze a target module.

        Preference order
        ----------------
        1) `head.freeze_target(module)` if provided and callable
        2) fallback implementation `_freeze_module_fallback`

        Parameters
        ----------
        module : nn.Module
            Target module to freeze.
        """
        fn = getattr(self.head, "freeze_target", None)
        if callable(fn):
            fn(module)
        else:
            self._freeze_module_fallback(module)

    # ---------------------------------------------------------------------
    # Target update primitives
    # ---------------------------------------------------------------------
    @staticmethod
    @th.no_grad()
    def _hard_update_fallback(target: nn.Module, source: nn.Module) -> None:
        """
        Fallback hard update via `state_dict` copy.

        Parameters
        ----------
        target : nn.Module
            Target network (updated in-place).
        source : nn.Module
            Source/online network (copied from).
        """
        target.load_state_dict(source.state_dict())

    @staticmethod
    @th.no_grad()
    def _soft_update_fallback(target: nn.Module, source: nn.Module, tau: float) -> None:
        """
        Fallback soft update (Polyak averaging).

        Parameters
        ----------
        target : nn.Module
            Target network (updated in-place).
        source : nn.Module
            Source/online network (copied from).
        tau : float
            Polyak factor in (0, 1]. Larger values update targets more aggressively.

        Notes
        -----
        Implements:
            θ_target <- (1 - tau) * θ_target + tau * θ_source
        """
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)

    # ---------------------------------------------------------------------
    # Target update controller
    # ---------------------------------------------------------------------
    def _maybe_update_target(
        self,
        *,
        target: Optional[nn.Module],
        source: nn.Module,
        interval: int,
        tau: float,
    ) -> None:
        """
        Conditionally update a target network.

        Parameters
        ----------
        target : Optional[nn.Module]
            Target network. If None, this function is a no-op.

        source : nn.Module
            Source (online) network.

        interval : int
            Update frequency in **core update calls**.
            - interval <= 0 : disabled
            - otherwise     : update when `(update_calls % interval) == 0`

            Notes
            -----
            This method updates on call 0 as well (i.e., when update_calls == 0).

        tau : float
            Update mode selector:
            - tau <= 0 : hard update
            - tau > 0  : soft update with Polyak factor tau (must be in (0, 1])

        Notes
        -----
        - The target is always frozen after updating (via `_freeze_target`).
        - Custom update logic may be provided by `head.hard_update` or `head.soft_update`.
        - Neither hard nor soft update functions are assumed to freeze targets.
        """
        if target is None:
            return

        interval_i = int(interval)
        if interval_i <= 0:
            return

        if (self._update_calls % interval_i) != 0:
            return

        tau_f = float(tau)
        if not (0.0 <= tau_f <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {tau_f}")

        if tau_f > 0.0:
            fn = getattr(self.head, "soft_update", None)
            if callable(fn):
                fn(target, source, tau=tau_f)
            else:
                self._soft_update_fallback(target, source, tau_f)
        else:
            fn = getattr(self.head, "hard_update", None)
            if callable(fn):
                fn(target, source)
            else:
                self._hard_update_fallback(target, source)

        self._freeze_target(target)

    # ---------------------------------------------------------------------
    # Optimizer / scheduler persistence helpers
    # ---------------------------------------------------------------------
    def _save_opt_sched(self, opt: Optimizer, sched: Optional[Any]) -> Dict[str, Any]:
        """
        Serialize optimizer and scheduler state.

        Parameters
        ----------
        opt : torch.optim.Optimizer
            Optimizer instance to serialize.
        sched : Optional[Any]
            Scheduler instance (or None). If None, scheduler state is stored as {}.

        Returns
        -------
        state : Dict[str, Any]
            Dictionary with keys:
            - "opt": optimizer_state_dict(opt)
            - "sched": scheduler_state_dict(sched) or {}
        """
        return {
            "opt": optimizer_state_dict(opt),
            "sched": scheduler_state_dict(sched) if sched is not None else {},
        }

    def _load_opt_sched(self, opt: Optimizer, sched: Optional[Any], state: Mapping[str, Any]) -> None:
        """
        Restore optimizer and scheduler state.

        Parameters
        ----------
        opt : torch.optim.Optimizer
            Optimizer instance to restore into.
        sched : Optional[Any]
            Scheduler instance to restore into (or None).
        state : Mapping[str, Any]
            Dict produced by `_save_opt_sched`.

        Notes
        -----
        - If `sched` is None, scheduler state is ignored.
        """
        opt_state = state.get("opt", None)
        if opt_state is not None:
            load_optimizer_state_dict(opt, opt_state)

        if sched is not None:
            load_scheduler_state_dict(sched, state.get("sched", {}))

    # ---------------------------------------------------------------------
    # Default persistence (core-wide)
    # ---------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Return serializable core state.

        Returns
        -------
        state : Dict[str, Any]
            Core state dictionary containing at least:
            - "update_calls": int
        """
        return {"update_calls": int(self._update_calls)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Load core state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dict produced by `state_dict()`.
        """
        self._update_calls = int(state.get("update_calls", 0))

    # ---------------------------------------------------------------------
    # Main contract
    # ---------------------------------------------------------------------
    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Run one update step from a batch and return metrics.

        Parameters
        ----------
        batch : Any
            Training batch. The structure is algorithm-specific (e.g., dict of
            tensors, namedtuple, replay sample, etc.).

        Returns
        -------
        metrics : Dict[str, float]
            Scalar diagnostics (losses, KL, entropy, gradient norms, etc.).
        """
        raise NotImplementedError


class ActorCriticCore(BaseCore, ABC):
    """
    Base class for actor-critic update engines.

    This core owns two independent optimization pipelines:
    - actor optimizer + (optional) scheduler
    - critic optimizer + (optional) scheduler

    It also implements checkpoint persistence for both pipelines.

    Required head interface (duck-typed)
    ------------------------------------
    - head.actor : nn.Module
        Policy / actor network.
    - head.critic : nn.Module
        Value function / critic network.

    Notes
    -----
    - KFAC support:
      Your `build_optimizer` enforces `name="kfac"` requires `model=...`. This
      class provides a helper that auto-injects `model=module` unless explicitly
      provided in `actor_optim_kwargs` / `critic_optim_kwargs`.
    """

    def __init__(
        self,
        *,
        head: Any,
        use_amp: bool = False,
        # optimizer
        actor_optim_name: str = "adamw",
        actor_lr: float = 3e-4,
        actor_weight_decay: float = 0.0,
        critic_optim_name: str = "adamw",
        critic_lr: float = 3e-4,
        critic_weight_decay: float = 0.0,
        # optimizer extra kwargs (e.g., KFAC knobs)
        actor_optim_kwargs: Optional[Mapping[str, Any]] = None,
        critic_optim_kwargs: Optional[Mapping[str, Any]] = None,
        # scheduler
        actor_sched_name: str = "none",
        critic_sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        super().__init__(head=head, use_amp=use_amp)

        if not hasattr(self.head, "actor") or not isinstance(self.head.actor, nn.Module):
            raise ValueError("ActorCriticCore requires head.actor: nn.Module")
        if not hasattr(self.head, "critic") or not isinstance(self.head.critic, nn.Module):
            raise ValueError("ActorCriticCore requires head.critic: nn.Module")

        actor_optim_kwargs_d = dict(actor_optim_kwargs or {})
        critic_optim_kwargs_d = dict(critic_optim_kwargs or {})

        # ---------------------------------------------------------------------
        # Optimizers
        # ---------------------------------------------------------------------
        self.actor_opt = self._build_optimizer_with_optional_model(
            module=self.head.actor,
            name=str(actor_optim_name),
            lr=float(actor_lr),
            weight_decay=float(actor_weight_decay),
            extra_kwargs=actor_optim_kwargs_d,
        )
        self.critic_opt = self._build_optimizer_with_optional_model(
            module=self.head.critic,
            name=str(critic_optim_name),
            lr=float(critic_lr),
            weight_decay=float(critic_weight_decay),
            extra_kwargs=critic_optim_kwargs_d,
        )

        # ---------------------------------------------------------------------
        # Schedulers
        # ---------------------------------------------------------------------
        ms = tuple(int(m) for m in milestones)
        self.actor_sched = build_scheduler(
            self.actor_opt,
            name=str(actor_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=ms,
        )
        self.critic_sched = build_scheduler(
            self.critic_opt,
            name=str(critic_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=ms,
        )

    @staticmethod
    def _build_optimizer_with_optional_model(
        *,
        module: nn.Module,
        name: str,
        lr: float,
        weight_decay: float,
        extra_kwargs: Dict[str, Any],
    ) -> Optimizer:
        """
        Build an optimizer and auto-inject `model=module` for KFAC.

        Parameters
        ----------
        module : nn.Module
            Module whose parameters will be optimized.
        name : str
            Optimizer name passed to `build_optimizer`.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay coefficient.
        extra_kwargs : Dict[str, Any]
            Extra optimizer-specific kwargs. If `name` is KFAC and `model` is not
            provided, this function inserts `model=module`.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            Instantiated optimizer.

        Notes
        -----
        Your `build_optimizer` enforces:
            build_optimizer(..., name="kfac", model=...) is required

        This helper ensures callers can simply request "kfac" without manually
        providing `model` every time.
        """
        opt_name = str(name).lower().strip().replace("-", "").replace("_", "")
        if opt_name == "kfac" and "model" not in extra_kwargs:
            extra_kwargs["model"] = module

        return build_optimizer(
            module.parameters(),
            name=str(name),
            lr=float(lr),
            weight_decay=float(weight_decay),
            **extra_kwargs,
        )

    def _step_scheds(self) -> None:
        """
        Step actor/critic schedulers if they exist.

        Notes
        -----
        This assumes you are using step-based schedulers (called once per optimizer
        step). If you want epoch-based semantics, call this at epoch boundaries.
        """
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state plus actor/critic optimizer + scheduler states.

        Returns
        -------
        state : Dict[str, Any]
            Includes:
            - base core state (update_calls)
            - "actor": {"opt": ..., "sched": ...}
            - "critic": {"opt": ..., "sched": ...}
        """
        s = super().state_dict()
        s.update(
            {
                "actor": self._save_opt_sched(self.actor_opt, self.actor_sched),
                "critic": self._save_opt_sched(self.critic_opt, self.critic_sched),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state plus actor/critic optimizer + scheduler states.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dict produced by `state_dict()`.
        """
        super().load_state_dict(state)
        if "actor" in state:
            self._load_opt_sched(self.actor_opt, self.actor_sched, state["actor"])
        if "critic" in state:
            self._load_opt_sched(self.critic_opt, self.critic_sched, state["critic"])

    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Algorithm-specific actor-critic update.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar diagnostics.
        """
        raise NotImplementedError


class QLearningCore(BaseCore, ABC):
    """
    Base class for Q-learning update engines (DQN family).

    This core owns:
    - Q-network optimizer + (optional) scheduler
    - persistence for that pipeline

    Required head interface (duck-typed)
    ------------------------------------
    - head.q : nn.Module
        Q-network (online network).
    """

    def __init__(
        self,
        *,
        head: Any,
        use_amp: bool = False,
        # optimizer
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        # scheduler
        sched_name: str = "none",
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        super().__init__(head=head, use_amp=use_amp)

        if not hasattr(self.head, "q") or not isinstance(self.head.q, nn.Module):
            raise ValueError("QLearningCore requires head.q: nn.Module")

        self.opt = build_optimizer(
            self.head.q.parameters(),
            name=str(optim_name),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.sched = build_scheduler(
            self.opt,
            name=str(sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
        )

    def _step_sched(self) -> None:
        """
        Step scheduler if it exists.

        Notes
        -----
        This assumes step-based semantics (called once per optimizer step).
        """
        if self.sched is not None:
            self.sched.step()

    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state plus optimizer/scheduler state.

        Returns
        -------
        state : Dict[str, Any]
            Includes:
            - base core state (update_calls)
            - "q": {"opt": ..., "sched": ...}
        """
        s = super().state_dict()
        s.update({"q": self._save_opt_sched(self.opt, self.sched)})
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore core state plus optimizer/scheduler state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State dict produced by `state_dict()`.
        """
        super().load_state_dict(state)
        if "q" in state:
            self._load_opt_sched(self.opt, self.sched, state["q"])

    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Algorithm-specific Q-learning update.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar diagnostics.
        """
        raise NotImplementedError
