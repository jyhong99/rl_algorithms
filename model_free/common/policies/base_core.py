from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence

import torch as th
import torch.nn as nn

from model_free.common.optimizers.optimizer_builder import (
    clip_grad_norm,
    optimizer_state_dict,
    load_optimizer_state_dict,
    build_optimizer,
)
from model_free.common.optimizers.scheduler_builder import (
    scheduler_state_dict,
    load_scheduler_state_dict,
    build_scheduler,
)


class BaseCore(ABC):
    """
    Base class for update engines ("cores").

    This class provides shared infrastructure:
      - `head` reference and normalized `device`
      - optional AMP GradScaler
      - update-call counter (used for scheduling target updates)
      - gradient clipping helper
      - generic target-network update helper (hard/soft) + freezing
      - optimizer/scheduler persistence helpers

    Notes
    -----
    - This class assumes `head` is a duck-typed container that may provide:
        * device: torch.device or str
        * freeze_target(module): optional
        * hard_update(target, source): optional
        * soft_update(target, source, tau=...): optional
    - Concrete cores must implement `update_from_batch`.
    """

    def __init__(self, *, head: Any, use_amp: bool = False) -> None:
        self.head = head

        # Normalize device
        dev = getattr(head, "device", th.device("cpu"))
        self.device = dev if isinstance(dev, th.device) else th.device(str(dev))

        # AMP is meaningful only on CUDA
        self.use_amp = bool(use_amp) and (self.device.type == "cuda") and th.cuda.is_available()

        # GradScaler creation (safe across torch variants)
        try:
            self.scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)
        except Exception:
            # fallback (rare torch builds)
            self.scaler = th.amp.GradScaler(enabled=self.use_amp)  # type: ignore[attr-defined]

        self._update_calls: int = 0

    # ---------------------------------------------------------------------
    # Bookkeeping
    # ---------------------------------------------------------------------
    @property
    def update_calls(self) -> int:
        """Number of times the core performed an update step."""
        return int(self._update_calls)

    def _bump(self) -> None:
        """Increment internal update counter."""
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
        Clip gradients in-place.

        Parameters
        ----------
        params : Any
            Iterable of parameters (e.g., module.parameters()).
        max_grad_norm : float
            If <= 0, clipping is disabled.
        optimizer : Optional[Any], optional
            Optimizer instance, needed by some AMP-unscale implementations.

        Notes
        -----
        - If AMP is enabled, gradients are unscaled before clipping.
        - `clip_grad_norm` is expected to be your project utility that supports
          scaler/optimizer arguments.
        """
        mg = float(max_grad_norm)
        if mg <= 0.0:
            return

        if self.use_amp:
            clip_grad_norm(params, mg, scaler=self.scaler, optimizer=optimizer)
        else:
            clip_grad_norm(params, mg)

    # ---------------------------------------------------------------------
    # Target freezing
    # ---------------------------------------------------------------------
    @staticmethod
    def _freeze_module_fallback(module: nn.Module) -> None:
        """Fallback freezing: requires_grad=False and eval()."""
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    def _freeze_target(self, module: nn.Module) -> None:
        """
        Freeze a target module.

        Preference order:
          1) head.freeze_target(module) if provided
          2) fallback implementation
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
        """Fallback hard update via state_dict copy."""
        target.load_state_dict(source.state_dict())

    @staticmethod
    @th.no_grad()
    def _soft_update_fallback(target: nn.Module, source: nn.Module, tau: float) -> None:
        """Fallback Polyak averaging."""
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)

    # ---------------------------------------------------------------------
    # Target update (generic)
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
            Target network. If None, no-op.
        source : nn.Module
            Online/source network.
        interval : int
            Update frequency in *core update calls*.
            - interval <= 0 : disabled
            - otherwise     : update when (update_calls % interval) == 0
        tau : float
            Update mode:
            - tau <= 0 : hard update
            - tau > 0  : soft update with Polyak factor tau, must be in (0,1]

        Notes
        -----
        - This method always freezes the target after updating.
        - Neither hard nor soft update functions are assumed to freeze.
        """
        if target is None:
            return

        interval = int(interval)
        if interval <= 0:
            return

        # Update on call 0, interval, 2*interval, ...
        if (self._update_calls % interval) != 0:
            return

        tau = float(tau)
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {tau}")

        if tau > 0.0:
            fn = getattr(self.head, "soft_update", None)
            if callable(fn):
                fn(target, source, tau=tau)
            else:
                self._soft_update_fallback(target, source, tau)
        else:
            fn = getattr(self.head, "hard_update", None)
            if callable(fn):
                fn(target, source)
            else:
                self._hard_update_fallback(target, source)

        self._freeze_target(target)

    # ---------------------------------------------------------------------
    # Optimizer / Scheduler persistence helpers
    # ---------------------------------------------------------------------
    def _save_opt_sched(self, opt: Any, sched: Any) -> Dict[str, Any]:
        """
        Serialize optimizer and scheduler state.

        Notes
        -----
        - `sched` may be None (e.g., sched_name="none"). In that case, stores {}.
        """
        return {
            "opt": optimizer_state_dict(opt),
            "sched": scheduler_state_dict(sched) if sched is not None else {},
        }

    def _load_opt_sched(self, opt: Any, sched: Any, state: Mapping[str, Any]) -> None:
        """
        Restore optimizer and scheduler state.

        Notes
        -----
        - If scheduler is None, scheduler state is ignored.
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
        """Return serializable core state."""
        return {"update_calls": int(self._update_calls)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load core state."""
        self._update_calls = int(state.get("update_calls", 0))

    # ---------------------------------------------------------------------
    # Main contract
    # ---------------------------------------------------------------------
    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """
        Run one update step from a batch and return metrics.

        Returns
        -------
        metrics : Dict[str, float]
            Scalar training diagnostics (losses, KL, entropy, grad norms, etc.).
        """
        raise NotImplementedError


class ActorCriticCore(BaseCore, ABC):
    """
    Base class for actor-critic update engines.

    Owns:
      - actor optimizer/scheduler
      - critic optimizer/scheduler
      - persistence for both

    Required head interface (duck-typed)
    ------------------------------------
    - head.actor: nn.Module
    - head.critic: nn.Module
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
        # NEW: optimizer extra kwargs (e.g., KFAC knobs)
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

        actor_optim_kwargs = dict(actor_optim_kwargs or {})
        critic_optim_kwargs = dict(critic_optim_kwargs or {})

        # ---------------------------------------------------------------------
        # Optimizers
        # ---------------------------------------------------------------------
        # IMPORTANT:
        # - build_optimizer(..., name='kfac', model=...) is required in your builder.
        # - We auto-inject model=... for kfac unless user explicitly provided one.
        self.actor_opt = self._build_optimizer_with_optional_model(
            module=self.head.actor,
            name=str(actor_optim_name),
            lr=float(actor_lr),
            weight_decay=float(actor_weight_decay),
            extra_kwargs=actor_optim_kwargs,
        )
        self.critic_opt = self._build_optimizer_with_optional_model(
            module=self.head.critic,
            name=str(critic_optim_name),
            lr=float(critic_lr),
            weight_decay=float(critic_weight_decay),
            extra_kwargs=critic_optim_kwargs,
        )

        # ---------------------------------------------------------------------
        # Schedulers
        # ---------------------------------------------------------------------
        self.actor_sched = build_scheduler(
            self.actor_opt,
            name=str(actor_sched_name),
            total_steps=int(total_steps),
            warmup_steps=int(warmup_steps),
            min_lr_ratio=float(min_lr_ratio),
            poly_power=float(poly_power),
            step_size=int(step_size),
            gamma=float(sched_gamma),
            milestones=tuple(int(m) for m in milestones),
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
            milestones=tuple(int(m) for m in milestones),
        )

    @staticmethod
    def _build_optimizer_with_optional_model(
        *,
        module: nn.Module,
        name: str,
        lr: float,
        weight_decay: float,
        extra_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Build optimizer and auto-inject `model=module` for KFAC.

        Rationale
        ---------
        Your build_optimizer enforces:
          build_optimizer(..., name='kfac', model=...) is required.

        This helper guarantees that requirement is satisfied without forcing
        every caller to remember passing model explicitly.
        """
        n = str(name).lower()

        # If user already provided model, respect it. Otherwise inject for KFAC.
        if n == "kfac" and "model" not in extra_kwargs:
            extra_kwargs["model"] = module

        return build_optimizer(
            module.parameters(),
            name=str(name),
            lr=float(lr),
            weight_decay=float(weight_decay),
            **extra_kwargs,
        )

    def _step_scheds(self) -> None:
        """Step actor/critic schedulers if they exist."""
        if self.actor_sched is not None:
            self.actor_sched.step()
        if self.critic_sched is not None:
            self.critic_sched.step()

    def state_dict(self) -> Dict[str, Any]:
        """Serialize core + optimizer/scheduler states."""
        s = super().state_dict()
        s.update(
            {
                "actor": self._save_opt_sched(self.actor_opt, self.actor_sched),
                "critic": self._save_opt_sched(self.critic_opt, self.critic_sched),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore core + optimizer/scheduler states."""
        super().load_state_dict(state)
        if "actor" in state:
            self._load_opt_sched(self.actor_opt, self.actor_sched, state["actor"])
        if "critic" in state:
            self._load_opt_sched(self.critic_opt, self.critic_sched, state["critic"])

    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError


class QLearningCore(BaseCore, ABC):
    """
    Base class for Q-learning discrete update engines (DQN family).

    Owns:
      - Q optimizer/scheduler
      - persistence

    Required head interface (duck-typed)
    ------------------------------------
    - head.q: nn.Module
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
        """Step scheduler if it exists."""
        if self.sched is not None:
            self.sched.step()

    def state_dict(self) -> Dict[str, Any]:
        """Serialize core + optimizer/scheduler states."""
        s = super().state_dict()
        s.update({"q": self._save_opt_sched(self.opt, self.sched)})
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore core + optimizer/scheduler states."""
        super().load_state_dict(state)
        if "q" in state:
            self._load_opt_sched(self.opt, self.sched, state["q"])

    @abstractmethod
    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError