from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

from .lion import Lion
from .kfac import KFAC


# =============================================================================
# Optimizer factory utilities
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
    Construct a PyTorch optimizer by name.

    This function centralizes optimizer creation for experiments, allowing you to
    switch optimizer types via a string identifier while keeping common hyperparameters
    in a single place.

    Supported optimizers
    --------------------
    - Adam:      ``torch.optim.Adam``
    - AdamW:     ``torch.optim.AdamW``
    - SGD:       ``torch.optim.SGD``
    - RMSprop:   ``torch.optim.RMSprop``
    - RAdam:     ``torch.optim.RAdam``
    - Lion:      custom ``Lion`` (sign-based momentum descent)
    - KFAC:      custom ``KFAC`` (Kronecker-factored curvature preconditioner)

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter] or Iterable[Dict[str, Any]]
        Parameters to optimize.

        Two input formats are supported:

        1) Flat parameters
           An iterable of ``nn.Parameter`` where all parameters share the same
           base hyperparameters.

        2) Parameter groups
           An iterable of PyTorch-compatible param-group dicts. Each dict must
           contain a ``"params"`` entry and may override group-wise hyperparameters
           (e.g., ``{"params": ..., "lr": ..., "weight_decay": ...}``).

        Notes
        -----
        For ``name="kfac"``:
            KFAC must attach hooks to modules in `model` and internally uses
            ``model.parameters()`` as the true optimized parameters. In that case,
            `params` is accepted for API consistency but effectively ignored.

    name : str, default="adamw"
        Optimizer identifier (case-insensitive). Common separators are normalized
        (e.g., "adam-weight-decay" -> "adamw").

    lr : float, default=3e-4
        Base learning rate. Must be > 0.

    weight_decay : float, default=0.0
        Weight decay coefficient. For AdamW/Lion this is typically treated as
        decoupled decay by the underlying optimizer implementation. Must be >= 0.

    betas : Tuple[float, float], default=(0.9, 0.999)
        Adam-like betas used for Adam/AdamW/RAdam. Each must be in [0, 1).

    eps : float, default=1e-8
        Numerical stability epsilon used by Adam/AdamW/RMSprop/RAdam. Must be > 0.

    momentum : float, default=0.0
        Momentum for SGD/RMSprop and the internal SGD used by KFAC.

    nesterov : bool, default=False
        Whether to enable Nesterov momentum for SGD.

    alpha : float, default=0.99
        Smoothing constant for RMSprop.

    centered : bool, default=False
        If True, uses centered RMSprop.

    lion_betas : Tuple[float, float], default=(0.9, 0.99)
        Lion-specific betas (beta1, beta2). Each must be in [0, 1).

    model : Optional[nn.Module], default=None
        Required only when ``name == "kfac"``. Used to register hooks and collect
        curvature statistics.

    damping : float, default=1e-2
        KFAC damping term. Must be >= 0.

    kfac_eps : float, default=0.95
        EMA coefficient for KFAC running covariances. Must be in (0, 1).

    Ts : int, default=1
        KFAC statistics collection interval (in optimizer steps). Must be > 0.

    Tf : int, default=10
        KFAC inverse update interval (in optimizer steps). Must be > 0.

    max_lr : float, default=1.0
        Upper bound on KFAC trust-region scaling factor. Must be > 0.

    trust_region : float, default=2e-3
        KFAC trust-region radius. Must be > 0.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Instantiated optimizer object.

    Raises
    ------
    ValueError
        If the optimizer `name` is unknown, required inputs are missing (e.g. KFAC
        without `model`), or hyperparameters are invalid.
    """
    # -------------------------
    # Basic validation
    # -------------------------
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

    # -------------------------
    # Normalize names (common variants)
    # -------------------------
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

    # -------------------------
    # Dispatch
    # -------------------------
    if opt == "adam":
        return optim.Adam(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)

    if opt == "adamw":
        return optim.AdamW(params, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)

    if opt == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            nesterov=bool(nesterov),
        )

    if opt == "rmsprop":
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=float(alpha),
            eps=eps,
            weight_decay=float(weight_decay),
            momentum=float(momentum),
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

        # Important: KFAC uses `model.parameters()` (hooked modules), not `params`.
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
    Build PyTorch optimizer parameter groups with practical defaults.

    This helper creates param groups that:
    - apply weight decay to "normal" weights,
    - avoid weight decay for biases and normalization parameters,
    - optionally apply prefix-based overrides for specific submodules.

    Parameters
    ----------
    named_params : Iterable[Tuple[str, nn.Parameter]]
        Typically ``model.named_parameters()``.

        Parameters with ``requires_grad=False`` are skipped.

    base_lr : float
        Default learning rate assigned to all groups unless overridden.

    base_weight_decay : float, default=0.0
        Default weight decay applied to decay-enabled parameters.

    no_decay_keywords : Sequence[str], default=("bias", "bn", "ln", "norm")
        Substring tokens that mark parameters as "no decay" (weight_decay=0.0).
        Matching is done on the lowercased parameter name.

        Examples
        --------
        - "bias" will match "fc.bias"
        - "bn" may match "encoder.bn1.weight"

        Caveat
        ------
        Substring matching can be over-inclusive depending on naming conventions.
        If you want stricter control, use longer tokens (e.g., "batchnorm") or
        use `overrides`.

    overrides : Optional[Sequence[Tuple[str, Dict[str, Any]]]], default=None
        Optional prefix-based overrides. Each item is:

            (prefix, config_dict)

        The first matching prefix wins. Prefixes are sorted by descending
        prefix length to reduce accidental partial matches.

        Example
        -------
        overrides = [
            ("actor.",  {"lr": 3e-4}),
            ("critic.", {"lr": 1e-3, "weight_decay": 0.0}),
        ]

        This sends all parameters whose names start with "actor." into a dedicated
        group with an overridden learning rate.

    Returns
    -------
    groups : List[Dict[str, Any]]
        List of param-group dicts compatible with torch.optim.

    Raises
    ------
    ValueError
        If base hyperparameters are invalid.
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got: {base_lr}")
    if base_weight_decay < 0:
        raise ValueError(f"base_weight_decay must be >= 0, got: {base_weight_decay}")

    overrides_list: List[Tuple[str, Dict[str, Any]]] = list(overrides) if overrides is not None else []
    overrides_list.sort(key=lambda x: len(x[0]), reverse=True)  # longer prefixes first

    decay_params: List[nn.Parameter] = []
    nodecay_params: List[nn.Parameter] = []

    # Collect params for override groups (prefix -> list[Parameter])
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

    for name, p in named_params:
        if not p.requires_grad:
            continue

        pfx = _match_override(name)
        if pfx is not None:
            override_bins[pfx].append(p)
            continue

        if _is_no_decay(name):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    groups: List[Dict[str, Any]] = []

    # Default groups (non-overridden)
    if decay_params:
        groups.append({"params": decay_params, "lr": float(base_lr), "weight_decay": float(base_weight_decay)})
    if nodecay_params:
        groups.append({"params": nodecay_params, "lr": float(base_lr), "weight_decay": 0.0})

    # Override groups
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

    This wrapper optionally unscales gradients via GradScaler before applying
    ``torch.nn.utils.clip_grad_norm_``.

    Parameters
    ----------
    parameters : Iterable[nn.Parameter]
        Parameters whose gradients will be clipped.

        Notes
        -----
        This function materializes the iterable into a list to avoid subtle issues
        when a generator is passed (generators can be exhausted by earlier passes).

    max_norm : float
        Maximum allowed gradient norm. If ``max_norm <= 0``, this is a no-op and
        the function returns 0.0.

    norm_type : float, default=2.0
        Norm type for computing the total norm (p-norm).

    scaler : Optional[torch.cuda.amp.GradScaler], default=None
        If provided, gradients are unscaled before clipping.

    optimizer : Optional[torch.optim.Optimizer], default=None
        Required if ``scaler`` is provided, because ``scaler.unscale_(optimizer)``
        needs the optimizer instance.

    Returns
    -------
    total_norm : float
        The total norm of the parameters' gradients *before* clipping, as reported
        by PyTorch.

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
    return float(total_norm.detach().cpu().item()) if th.is_tensor(total_norm) else float(total_norm)


def optimizer_state_dict(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Return a checkpoint-ready optimizer state dict.

    This is a thin wrapper around ``optimizer.state_dict()`` for consistency and
    potential future extension (e.g., filtering, compatibility transforms).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance.

    Returns
    -------
    state : Dict[str, Any]
        The optimizer state dict.
    """
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: Optimizer, state: Mapping[str, Any]) -> None:
    """
    Load optimizer state from a checkpoint.

    This is a thin wrapper around ``optimizer.load_state_dict(...)`` for
    consistency and potential future extension (e.g., mapping devices).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Target optimizer instance.
    state : Mapping[str, Any]
        State dict previously produced by `optimizer_state_dict` or directly by
        `optimizer.state_dict()`.

    Returns
    -------
    None
    """
    optimizer.load_state_dict(dict(state))
