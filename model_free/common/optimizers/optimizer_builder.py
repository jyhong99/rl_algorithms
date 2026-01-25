from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

from .lion import Lion
from .kfac import KFAC


# =============================================================================
# Optimizer factory
# =============================================================================
def build_optimizer(
    params: Union[Iterable[nn.Parameter], Iterable[Dict[str, Any]]],
    *,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.0,
    nesterov: bool = False,
    alpha: float = 0.99,
    centered: bool = False,
    # Lion
    lion_betas: Tuple[float, float] = (0.9, 0.99),
    # KFAC (requires model)
    model: Optional[nn.Module] = None,
    damping: float = 1e-2,
    kfac_eps: float = 0.95,
    Ts: int = 1,
    Tf: int = 10,
    max_lr: float = 1.0,
    trust_region: float = 2e-3,
) -> Optimizer:
    """
    Build a PyTorch optimizer.

    Parameters
    ----------
    params : Iterable[nn.Parameter] or Iterable[Dict[str, Any]]
        Parameters to optimize. Two formats are supported:

        1) Flat parameters:
           Iterable[nn.Parameter] where all parameters share the same base
           hyperparameters.

        2) Parameter groups:
           Iterable[Dict[str, Any]] compatible with PyTorch param_group format.
           Each dict must contain a "params" entry and may override lr,
           weight_decay, etc.

        Important:
        - For `name="kfac"`, this function will construct the optimizer using
          `model.parameters()` internally. In that case, `params` is accepted
          for API consistency, but the true optimized parameters are tied to
          `model`.

    name : str, optional
        Optimizer identifier (case-insensitive), by default "adamw".
        Supported:
        - "adam", "adamw", "sgd", "rmsprop", "radam", "lion", "kfac"

    lr : float, optional
        Base learning rate, by default 3e-4.

    weight_decay : float, optional
        Weight decay coefficient, by default 0.0.

    betas : Tuple[float, float], optional
        Adam-like betas for Adam/AdamW/RAdam, by default (0.9, 0.999).

    eps : float, optional
        Numerical stability epsilon for Adam/AdamW/RMSprop/RAdam, by default 1e-8.

    momentum : float, optional
        Momentum for SGD/RMSprop, and internal SGD of KFAC, by default 0.0.

    nesterov : bool, optional
        Nesterov momentum for SGD, by default False.

    alpha : float, optional
        RMSprop smoothing constant, by default 0.99.

    centered : bool, optional
        Whether to use centered RMSprop, by default False.

    lion_betas : Tuple[float, float], optional
        Betas for Lion, by default (0.9, 0.99).

    model : Optional[nn.Module], optional
        Required when name == "kfac". Used to register hooks and collect
        curvature statistics.

    damping : float, optional
        KFAC damping term, by default 1e-2.

    kfac_eps : float, optional
        KFAC Polyak/EMA coefficient for running covariances, by default 0.95.

    Ts : int, optional
        KFAC statistics collection interval, by default 1.

    Tf : int, optional
        KFAC inverse update interval, by default 10.

    max_lr : float, optional
        KFAC trust-region scaling upper bound, by default 1.0.

    trust_region : float, optional
        KFAC trust-region radius, by default 2e-3.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Instantiated optimizer.

    Raises
    ------
    ValueError
        If `name` is unknown, or if `name == "kfac"` but `model` is None,
        or if hyperparameters are invalid.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got: {lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got: {weight_decay}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got: {eps}")
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got: {momentum}")

    b1, b2 = float(betas[0]), float(betas[1])
    if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
        raise ValueError(f"betas must be in [0, 1), got: {betas}")

    # normalize names (handle common variants)
    opt = name.lower().strip().replace("-", "").replace("_", "")
    if opt in ("adamw", "adamweightdecay"):
        opt = "adamw"
    elif opt in ("adam",):
        opt = "adam"
    elif opt in ("sgd",):
        opt = "sgd"
    elif opt in ("rmsprop",):
        opt = "rmsprop"
    elif opt in ("radam",):
        opt = "radam"
    elif opt in ("lion",):
        opt = "lion"
    elif opt in ("kfac",):
        opt = "kfac"

    if opt == "adam":
        return optim.Adam(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)

    if opt == "adamw":
        return optim.AdamW(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)

    if opt == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=bool(nesterov))

    if opt == "rmsprop":
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=float(alpha),
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=bool(centered),
        )

    if opt == "radam":
        return optim.RAdam(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)

    if opt == "lion":
        lb1, lb2 = float(lion_betas[0]), float(lion_betas[1])
        if not (0.0 <= lb1 < 1.0 and 0.0 <= lb2 < 1.0):
            raise ValueError(f"lion_betas must be in [0, 1), got: {lion_betas}")
        return Lion(params, lr=lr, betas=(lb1, lb2), weight_decay=weight_decay)

    if opt == "kfac":
        if model is None:
            raise ValueError("build_optimizer(..., name='kfac', model=...) is required.")
        if damping < 0:
            raise ValueError(f"damping must be >= 0, got: {damping}")
        if not (0.0 < kfac_eps < 1.0):
            raise ValueError(f"kfac_eps must be in (0, 1), got: {kfac_eps}")
        if Ts <= 0 or Tf <= 0:
            raise ValueError(f"Ts and Tf must be > 0, got Ts={Ts}, Tf={Tf}")
        if max_lr <= 0:
            raise ValueError(f"max_lr must be > 0, got: {max_lr}")
        if trust_region <= 0:
            raise ValueError(f"trust_region must be > 0, got: {trust_region}")

        return KFAC(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            damping=damping,
            momentum=momentum,
            eps=kfac_eps,
            Ts=Ts,
            Tf=Tf,
            max_lr=max_lr,
            trust_region=trust_region,
        )

    raise ValueError(f"Unknown optimizer name: {name!r}")


def make_param_groups(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    *,
    base_lr: float,
    base_weight_decay: float = 0.0,
    no_decay_keywords: Sequence[str] = ("bias", "bn", "ln", "norm"),
    overrides: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build PyTorch optimizer param groups with practical defaults.

    Parameters
    ----------
    named_params : Iterable[Tuple[str, nn.Parameter]]
        Typically `model.named_parameters()`. Parameters with requires_grad=False
        are skipped.

    base_lr : float
        Default learning rate for all groups.

    base_weight_decay : float, optional
        Default weight decay for decay-enabled groups, by default 0.0.

    no_decay_keywords : Sequence[str], optional
        If parameter name contains any of these tokens (lowercased), it is assigned
        to a no-decay group (weight_decay=0.0), by default ("bias", "bn", "ln", "norm").

        Note:
        - Token matching is substring-based. For some naming conventions, "bn" might
          match unintended parameter names. If that is a concern, tighten tokens
          (e.g., ("bias", "batchnorm", "layernorm", "norm")) or use overrides.

    overrides : Optional[Sequence[Tuple[str, Dict[str, Any]]]], optional
        Optional prefix-based overrides. First match wins.
        Example:
            overrides = [
                ("actor.",  {"lr": 3e-4}),
                ("critic.", {"lr": 1e-3, "weight_decay": 0.0}),
            ]

        Implementation detail:
        - Prefixes are matched after sorting by descending prefix length to
          reduce accidental partial matches.

    Returns
    -------
    groups : List[Dict[str, Any]]
        List of param_group dicts compatible with torch.optim.
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got: {base_lr}")
    if base_weight_decay < 0:
        raise ValueError(f"base_weight_decay must be >= 0, got: {base_weight_decay}")

    overrides_list: List[Tuple[str, Dict[str, Any]]] = list(overrides) if overrides is not None else []
    # longer prefixes first => safer "first match wins"
    overrides_list.sort(key=lambda x: len(x[0]), reverse=True)

    decay_params: List[nn.Parameter] = []
    nodecay_params: List[nn.Parameter] = []

    override_bins: Dict[str, List[nn.Parameter]] = {pfx: [] for pfx, _ in overrides_list}
    override_cfgs: Dict[str, Dict[str, Any]] = {pfx: dict(cfg) for pfx, cfg in overrides_list}

    def _is_no_decay(name: str) -> bool:
        lname = name.lower()
        return any(k in lname for k in no_decay_keywords)

    def _match_override(name: str) -> Optional[str]:
        for pfx, _ in overrides_list:
            if name.startswith(pfx):
                return pfx
        return None

    for n, p in named_params:
        if not p.requires_grad:
            continue

        pfx = _match_override(n)
        if pfx is not None:
            override_bins[pfx].append(p)
            continue

        if _is_no_decay(n):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "lr": float(base_lr), "weight_decay": float(base_weight_decay)})
    if nodecay_params:
        groups.append({"params": nodecay_params, "lr": float(base_lr), "weight_decay": 0.0})

    for pfx, _ in overrides_list:
        ps = override_bins[pfx]
        if not ps:
            continue
        g: Dict[str, Any] = {"params": ps, "lr": float(base_lr), "weight_decay": float(base_weight_decay)}
        g.update(override_cfgs[pfx])
        groups.append(g)

    return groups


def clip_grad_norm(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    *,
    scaler: Optional[th.cuda.amp.GradScaler] = None,
    optimizer: Optional[Optimizer] = None,
) -> float:
    """
    AMP-safe gradient norm clipping.

    Parameters
    ----------
    parameters : Iterable[nn.Parameter]
        Parameters whose gradients will be clipped.

        Note:
        - This function materializes the iterable into a list to avoid issues when
          a generator is passed (generators can be exhausted by earlier passes).

    max_norm : float
        Maximum allowed norm. If <= 0, no-op and returns 0.0.

    norm_type : float, optional
        p-norm type, by default 2.0.

    scaler : Optional[torch.cuda.amp.GradScaler], optional
        If provided, gradients are unscaled before clipping.

    optimizer : Optional[torch.optim.Optimizer], optional
        Required if `scaler` is provided (used by scaler.unscale_).

    Returns
    -------
    total_norm : float
        Pre-clip total norm as reported by PyTorch.

    Raises
    ------
    ValueError
        If `scaler` is provided but `optimizer` is None.
    """
    if max_norm <= 0:
        return 0.0

    params_list = list(parameters)

    if scaler is not None:
        if optimizer is None:
            raise ValueError("When scaler is provided, optimizer must be provided too.")
        scaler.unscale_(optimizer)

    total_norm = nn.utils.clip_grad_norm_(params_list, max_norm, norm_type=float(norm_type))
    if th.is_tensor(total_norm):
        return float(total_norm.detach().cpu().item())
    return float(total_norm)


def optimizer_state_dict(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get a checkpoint-ready optimizer state dict.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    Returns
    -------
    state : Dict[str, Any]
        Optimizer state dict.
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: Optimizer, state: Mapping[str, Any]) -> None:
    """
    Load optimizer state dict from a checkpoint.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    state : Mapping[str, Any]

    Returns
    -------
    None
    """
    optimizer.load_state_dict(dict(state))