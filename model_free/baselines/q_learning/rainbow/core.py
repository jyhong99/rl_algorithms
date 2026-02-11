from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch as th

from model_free.common.utils.common_utils import _to_column, _to_scalar
from model_free.common.policies.base_core import QLearningCore
from model_free.common.utils.policy_utils import _get_per_weights, _distribution_projection


class RainbowCore(QLearningCore):
    """
    Rainbow (C51) update engine for discrete action spaces.

    This core implements the *distributional* Bellman update used by Rainbow's C51
    component. It reuses the shared :class:`~model_free.common.policies.base_core.QLearningCore`
    infrastructure for optimizer/scheduler wiring, AMP bookkeeping, gradient
    clipping, and target-network updates.

    Algorithm summary
    -----------------
    Let ``Z(s, a)`` be the return distribution for action ``a`` at state ``s``.
    In C51, ``Z(s, a)`` is represented as a categorical distribution over a fixed
    discrete support (atoms):

    - Support (atoms): ``z = linspace(v_min, v_max, K)``
    - Predicted PMF: ``p_theta(z_k | s, a)``, shape ``(K,)`` per (s, a)

    For a replay batch, this core performs:

    1) **Online distribution**:
       predict ``p_theta(z | s, a)`` for the *taken* action ``a``.

    2) **Target distribution**:
       choose greedy action ``a*`` at next state ``s'`` (Double DQN option),
       then take the target PMF ``p_bar(z | s', a*)`` from the target network.

    3) **Distributional Bellman backup + projection**:
       apply the distributional Bellman operator and project back onto the fixed
       support ``[v_min, v_max]``:

       ``Tz p_bar := Proj( r + gamma^n * (1-done) * z, p_bar(z|s',a*) )``

    4) **Loss**:
       minimize cross-entropy between projected target distribution and online
       predicted distribution:

       ``L = - E_i[ sum_k target_i[k] * log p_theta_i[k] ]``

    Reused from QLearningCore
    -------------------------
    - Optimizer / scheduler: ``self.opt`` / ``self.sched``
    - Update bookkeeping: ``self._bump()``, ``self.update_calls``
    - AMP scaler: ``self.scaler`` when ``use_amp=True``
    - Gradient clipping: ``self._clip_params(...)``
    - Target update helper: ``self._maybe_update_target(...)``

    Expected head contract (duck-typed)
    -----------------------------------
    The provided ``head`` must expose:

    - ``head.q``: nn.Module
        * ``head.q.dist(obs) -> (B, A, K)`` categorical PMFs over atoms
        * optional ``head.q.reset_noise()`` for NoisyNet
    - ``head.q_target``: nn.Module
        * ``head.q_target.dist(obs) -> (B, A, K)``
        * optional ``head.q_target.reset_noise()``
    - ``head.q_values(obs) -> (B, A)``
        Expected Q-values computed from distribution (used for action selection).
    - ``head.q_values_target(obs) -> (B, A)``
        Expected Q-values from target network.
    - ``head.support``: torch.Tensor, shape ``(K,)``
        Atom values (C51 support).
    - ``head.v_min`` / ``head.v_max``: float
        Support bounds.

    Notes
    -----
    - PER weighting:
        If the batch carries PER weights, ``w`` is read via :func:`_get_per_weights`
        and used to compute a weighted mean loss.
    - PER priority proxy:
        This implementation returns per-sample cross-entropy magnitude as
        ``"per/td_errors"``. It is not a true TD error, but provides a stable and
        monotone priority signal for PER buffers.
    """

    def __init__(
        self,
        *,
        head: Any,
        # ------------------------------------------------------------------
        # TD / target update
        # ------------------------------------------------------------------
        gamma: float = 0.99,
        n_step: int = 1,
        target_update_interval: int = 1000,
        tau: float = 0.0,
        double_dqn: bool = True,
        # ------------------------------------------------------------------
        # Gradient / AMP
        # ------------------------------------------------------------------
        max_grad_norm: float = 0.0,
        use_amp: bool = False,
        # ------------------------------------------------------------------
        # PER proxy / numeric stability
        # ------------------------------------------------------------------
        per_eps: float = 1e-6,
        log_eps: float = 1e-6,
        # ------------------------------------------------------------------
        # Optimizer / scheduler (QLearningCore)
        # ------------------------------------------------------------------
        optim_name: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        sched_name: str = "none",
        # scheduler shared knobs
        total_steps: int = 0,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        poly_power: float = 1.0,
        step_size: int = 1000,
        sched_gamma: float = 0.99,
        milestones: Sequence[int] = (),
    ) -> None:
        """
        Parameters
        ----------
        head : Any
            Head object satisfying the contract described in the class docstring.
        gamma : float, default=0.99
            Discount factor in ``[0, 1)``.
        n_step : int, default=1
            N-step return horizon. If ``n_step > 1`` and the replay batch provides
            n-step fields, the core will use them. The discount used in the backup
            is ``gamma**n_step``.
        target_update_interval : int, default=1000
            Target update period measured in **core update calls**:
            - ``0``: never update target network
            - ``1``: update every update call
        tau : float, default=0.0
            Polyak coefficient for soft updates performed at ``target_update_interval``.
            Convention:
            - ``tau <= 0``: treat as hard update (copy parameters)
            - ``tau > 0`` : soft update
        double_dqn : bool, default=True
            If True, select next greedy action using *online* expected Q-values and
            evaluate the distribution using the *target* network (Double DQN).
        max_grad_norm : float, default=0.0
            Global gradient norm clipping threshold. ``0`` disables clipping.
        use_amp : bool, default=False
            Enable mixed precision (AMP) on CUDA (uses QLearningCore scaler).
        per_eps : float, default=1e-6
            Small epsilon used when exporting PER priority proxy to avoid zeros.
        log_eps : float, default=1e-6
            Clamp floor for probabilities before ``log`` to prevent ``-inf``.

        Raises
        ------
        ValueError
            If any hyperparameter is out of its valid range.
        """
        super().__init__(
            head=head,
            use_amp=use_amp,
            optim_name=optim_name,
            lr=lr,
            weight_decay=weight_decay,
            sched_name=sched_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            poly_power=poly_power,
            step_size=step_size,
            sched_gamma=sched_gamma,
            milestones=milestones,
        )

        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.double_dqn = bool(double_dqn)

        self.max_grad_norm = float(max_grad_norm)
        self.per_eps = float(per_eps)
        self.log_eps = float(log_eps)

        # ------------------------------------------------------------------
        # Hyperparameter validation
        # ------------------------------------------------------------------
        if self.n_step <= 0:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0,1), got {self.gamma}")
        if self.target_update_interval < 0:
            raise ValueError(f"target_update_interval must be >= 0, got {self.target_update_interval}")
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0,1], got {self.tau}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.per_eps < 0.0:
            raise ValueError(f"per_eps must be >= 0, got {self.per_eps}")
        if self.log_eps <= 0.0:
            raise ValueError(f"log_eps must be > 0, got {self.log_eps}")

        # Freeze target net once at init (core owns this invariant).
        q_target = getattr(self.head, "q_target", None)
        if q_target is not None:
            # QLearningCore (or its base) is expected to provide a freezing helper.
            # If your actual base method name differs, rename accordingly.
            self._freeze_target(q_target)

    # =============================================================================
    # Helpers
    # =============================================================================
    def _get_nstep_fields(self, batch: Any) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Select reward/done/next_obs tensors for 1-step or n-step backups.

        This helper prefers n-step fields when:
        - ``self.n_step > 1``, and
        - all of the following batch attributes exist and are not None:
            * ``batch.n_step_returns``
            * ``batch.n_step_dones``
            * ``batch.n_step_next_observations``

        Otherwise it falls back to standard 1-step fields:
        - ``batch.rewards``, ``batch.dones``, ``batch.next_observations``

        Parameters
        ----------
        batch : Any
            Replay batch object that exposes the required attributes.

        Returns
        -------
        rewards : torch.Tensor
            Reward tensor, shape ``(B,)`` or ``(B,1)`` (caller normalizes).
        dones : torch.Tensor
            Done tensor, shape ``(B,)`` or ``(B,1)`` (caller normalizes).
        next_obs : torch.Tensor
            Next observation tensor, shape ``(B, obs_dim)``.
        """
        if int(self.n_step) <= 1:
            return (
                batch.rewards.to(self.device),
                batch.dones.to(self.device),
                batch.next_observations.to(self.device),
            )

        r = getattr(batch, "n_step_returns", None)
        d = getattr(batch, "n_step_dones", None)
        ns = getattr(batch, "n_step_next_observations", None)

        if (r is not None) and (d is not None) and (ns is not None):
            return (r.to(self.device), d.to(self.device), ns.to(self.device))

        return (
            batch.rewards.to(self.device),
            batch.dones.to(self.device),
            batch.next_observations.to(self.device),
        )

    def _gamma_n(self) -> float:
        """
        Compute ``gamma**n`` for the Bellman backup.

        Returns
        -------
        float
            Discount exponent used for the backup. If ``n_step`` is invalid
            (should not happen due to validation), it defaults to exponent 1.
        """
        return float(self.gamma) ** max(1, int(self.n_step))

    # =============================================================================
    # Update
    # =============================================================================
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Perform one Rainbow (C51) update from a replay batch.

        Parameters
        ----------
        batch : Any
            Replay batch providing at least:
            - ``observations``: (B, obs_dim)
            - ``actions``: (B,) or (B,1) integer action indices
            - ``rewards``: (B,) or (B,1)
            - ``next_observations``: (B, obs_dim)
            - ``dones``: (B,) or (B,1)

            Optionally, when ``n_step > 1``:
            - ``n_step_returns``, ``n_step_dones``, ``n_step_next_observations``

            Optionally for PER:
            - weight fields consumed by :func:`_get_per_weights`.

        Returns
        -------
        metrics : Dict[str, Any]
            Scalar logs plus PER priority proxy:
            - ``"per/td_errors"``: numpy array of shape (B,), where each element is
              the per-sample cross-entropy magnitude (clipped by ``per_eps``).
        """
        self._bump()

        # ------------------------------------------------------------------
        # Move batch to device
        # ------------------------------------------------------------------
        obs = batch.observations.to(self.device)          # (B, obs_dim)
        act = batch.actions.to(self.device).long()        # (B,) or (B,1)

        rew, done, nxt = self._get_nstep_fields(batch)
        rew = _to_column(rew)                             # (B,1)
        done = _to_column(done)                           # (B,1)

        B = int(obs.shape[0])
        w = _get_per_weights(batch, B, device=self.device)  # (B,1) or None

        # ------------------------------------------------------------------
        # Current distribution for taken action: p_theta(z | s, a)
        # ------------------------------------------------------------------
        dist_all = self.head.q.dist(obs)                  # (B, A, K)
        if dist_all.dim() != 3:
            raise ValueError(f"head.q.dist(obs) must be (B,A,K), got {tuple(dist_all.shape)}")

        K = int(dist_all.shape[-1])

        # gather along action dimension -> (B,1,K) -> (B,K)
        dist_a = dist_all.gather(1, act.view(-1, 1, 1).expand(-1, 1, K)).squeeze(1)  # (B, K)

        # ------------------------------------------------------------------
        # Build projected target distribution (no grad)
        # ------------------------------------------------------------------
        with th.no_grad():
            if self.double_dqn:
                # Optional Rainbow convention: resample online NoisyNet noise before greedy selection.
                if hasattr(self.head.q, "reset_noise"):
                    self.head.q.reset_noise()
                q_next_online = self.head.q_values(nxt)      # (B, A)
                a_star = th.argmax(q_next_online, dim=-1)    # (B,)
            else:
                q_next_t = self.head.q_values_target(nxt)    # (B, A)
                a_star = th.argmax(q_next_t, dim=-1)         # (B,)

            next_dist_all = self.head.q_target.dist(nxt)     # (B, A, K)
            if next_dist_all.dim() != 3:
                raise ValueError(
                    f"head.q_target.dist(nxt) must be (B,A,K), got {tuple(next_dist_all.shape)}"
                )
            if int(next_dist_all.shape[-1]) != K:
                raise ValueError(
                    f"Atom count mismatch: online K={K} vs target K={int(next_dist_all.shape[-1])}"
                )

            next_dist = next_dist_all.gather(
                1, a_star.view(-1, 1, 1).expand(-1, 1, K)
            ).squeeze(1)  # (B, K)

            target_dist = _distribution_projection(
                next_dist=next_dist,                    # (B, K)
                rewards=rew,                            # (B, 1)
                dones=done,                             # (B, 1)
                gamma=self._gamma_n(),                  # scalar gamma^n
                support=self.head.support,              # (K,)
                v_min=float(self.head.v_min),
                v_max=float(self.head.v_max),
            )  # (B, K)

        # ------------------------------------------------------------------
        # Cross-entropy loss: -sum_k target[k] * log p[k]
        # ------------------------------------------------------------------
        logp = th.log(dist_a.clamp(min=self.log_eps))        # (B, K)
        per_sample = -(target_dist * logp).sum(dim=-1)       # (B,)

        loss = per_sample.mean() if w is None else (per_sample.view(-1, 1) * w).mean()

        # ------------------------------------------------------------------
        # Optimizer step (online network only)
        # ------------------------------------------------------------------
        self.opt.zero_grad(set_to_none=True)

        if self.use_amp:
            with th.cuda.amp.autocast(enabled=True):
                loss_amp = loss
            self.scaler.scale(loss_amp).backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm, optimizer=self.opt)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_params(self.head.q.parameters(), max_grad_norm=self.max_grad_norm)
            self.opt.step()

        if self.sched is not None:
            self.sched.step()

        # ------------------------------------------------------------------
        # Target update (hard/soft)
        # ------------------------------------------------------------------
        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        # ------------------------------------------------------------------
        # Metrics + PER proxy vector
        # ------------------------------------------------------------------
        with th.no_grad():
            support = self.head.support.to(self.device)
            if support.dim() != 1 or int(support.shape[0]) != K:
                raise ValueError(f"head.support must be (K,), got {tuple(support.shape)} vs K={K}")

            # Expected Q(s,a) from categorical distribution.
            q_taken = (dist_a * support.view(1, -1)).sum(dim=-1)  # (B,)

            # PER priority proxy: cross-entropy magnitude.
            td_abs = per_sample.detach().abs().clamp(min=self.per_eps)  # (B,)

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_taken.mean())),
            "target/mean": float(_to_scalar(target_dist.mean())),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "per/td_errors": td_abs.detach().cpu().numpy(),
        }

    # =============================================================================
    # Persistence
    # =============================================================================
    def state_dict(self) -> Dict[str, Any]:
        """
        Serialize core state (optimizer/scheduler/update counter + hyperparameters).

        Notes
        -----
        The base :class:`QLearningCore` state typically includes:
        - ``update_calls``
        - optimizer state (and scheduler state if present)

        This method adds Rainbow/C51-specific hyperparameters for reproducibility.
        """
        s = super().state_dict()
        s.update(
            {
                "gamma": float(self.gamma),
                "n_step": int(self.n_step),
                "target_update_interval": int(self.target_update_interval),
                "tau": float(self.tau),
                "double_dqn": bool(self.double_dqn),
                "max_grad_norm": float(self.max_grad_norm),
                "per_eps": float(self.per_eps),
                "log_eps": float(self.log_eps),
            }
        )
        return s

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Restore inherited state (optimizer/scheduler/update counter).

        Notes
        -----
        Hyperparameters are constructor-owned by design and are **not** overridden
        from the checkpoint to avoid silent behavior changes.
        """
        super().load_state_dict(state)
        return
