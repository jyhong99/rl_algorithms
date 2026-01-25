from __future__ import annotations

import os
import sys

from typing import Any, Callable, List, Tuple

import torch as th
import torch.nn as nn

def _bootstrap_sys_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        # If parent contains a "model_free" package dir, add parent to sys.path
        if os.path.isdir(os.path.join(parent, "model_free")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        cur = parent

    # Fallback: add grandparent (often works when tests/ is inside package)
    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_bootstrap_sys_path()

from model_free.common.testers.test_utils import (
    seed_all,
    run_tests,
    assert_eq,
    assert_true,
    assert_raises
)

from model_free.common.optimizers.kfac import KFAC
from model_free.common.optimizers.lion import Lion
from model_free.common.optimizers.optimizer_builder import (
    build_optimizer,
    make_param_groups,
    clip_grad_norm,
)
# scheduler
from model_free.common.optimizers.scheduler_builder import build_scheduler
from model_free.common.testers.test_harness import TinyCNN, TinyMLP

# =============================================================================
# Tests: Lion
# =============================================================================
def test_lion_step_updates_params_and_creates_state():
    seed_all(0)
    model = TinyMLP()
    opt = Lion(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)

    x = th.randn(32, 8)
    y = th.randn(32, 4)

    p0 = [p.detach().clone() for p in model.parameters()]

    pred = model(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()

    opt.step()

    # params should change
    changed = False
    for p, q in zip(model.parameters(), p0):
        if not th.allclose(p.detach(), q):
            changed = True
            break
    assert_true(changed, "Lion step did not update parameters")

    # state should contain exp_avg for at least one param
    any_state = any(("exp_avg" in opt.state[p]) for p in opt.state.keys())
    assert_true(any_state, "Lion state did not create exp_avg buffers")


def test_lion_invalid_hparams():
    model = TinyMLP()
    assert_raises(ValueError, lambda: Lion(model.parameters(), lr=0.0))
    assert_raises(ValueError, lambda: Lion(model.parameters(), betas=(1.0, 0.9)))
    assert_raises(ValueError, lambda: Lion(model.parameters(), weight_decay=-1.0))


def test_lion_sparse_grad_raises():
    # Create a sparse parameter gradient by crafting a sparse tensor and assigning it.
    # Lion should raise RuntimeError when encountering p.grad.is_sparse.
    p = nn.Parameter(th.zeros(10))
    opt = Lion([p], lr=1e-3)

    idx = th.tensor([[1, 3, 7]])
    vals = th.tensor([1.0, -2.0, 0.5])
    sp = th.sparse_coo_tensor(idx, vals, size=(10,))
    p.grad = sp  # type: ignore

    assert_raises(RuntimeError, lambda: opt.step())


# =============================================================================
# Tests: KFAC stats collection & step
# =============================================================================
def _run_one_backward(model: nn.Module, *, batch: int = 16) -> float:
    model.zero_grad(set_to_none=True)
    if isinstance(model, TinyCNN):
        x = th.randn(batch, 3, 8, 8)
        y = th.randn(batch, 5)
    else:
        x = th.randn(batch, 8)
        y = th.randn(batch, 4)
    pred = model(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()
    return float(loss.detach().cpu().item())


def test_kfac_registers_layers_and_hooks():
    seed_all(0)
    model = TinyMLP()
    opt = KFAC(model, lr=0.1, Ts=1, Tf=2)

    # should find Linear layers
    assert_true(len(opt._trainable_layers) >= 2, "KFAC did not register expected trainable layers")
    assert_true(len(opt._hook_handles) >= 2 * len(opt._trainable_layers), "KFAC hook handles missing")


def test_kfac_aa_hat_updates_on_forward_pre_hook():
    seed_all(0)
    model = TinyMLP()
    opt = KFAC(model, lr=0.1, Ts=1, Tf=2)

    # run a forward/backward once (aa_hat captured on forward_pre_hook)
    _run_one_backward(model)

    # At least one layer should have aa_hat
    assert_true(len(opt._aa_hat) > 0, "KFAC aa_hat not collected")

    # Check shape for first Linear layer: (in_features+1, in_features+1) due to bias column
    first = None
    for m in opt._trainable_layers:
        if isinstance(m, nn.Linear):
            first = m
            break
    assert_true(first is not None, "No Linear layer found in trainable layers")

    aa = opt._aa_hat.get(first)
    assert_true(aa is not None, "aa_hat missing for first Linear layer")

    in_dim = first.in_features + 1  # bias appended
    assert_eq(tuple(aa.shape), (in_dim, in_dim), "aa_hat shape mismatch for Linear (bias-augmented)")


def test_kfac_gg_hat_gated_by_fisher_backprop():
    seed_all(0)
    model = TinyMLP()
    opt = KFAC(model, lr=0.1, Ts=1, Tf=2)

    # by default fisher_backprop False => gg_hat should remain empty after backward
    _run_one_backward(model)
    assert_eq(len(opt._gg_hat), 0, "gg_hat should be gated off when fisher_backprop=False")

    # enable fisher and run again => gg_hat should populate
    opt.set_fisher_backprop(True)
    _run_one_backward(model)
    assert_true(len(opt._gg_hat) > 0, "gg_hat not collected when fisher_backprop=True")


def test_kfac_step_updates_params_and_increments_k():
    seed_all(0)
    model = TinyMLP()
    opt = KFAC(model, lr=0.25, Ts=1, Tf=1, damping=1e-2)

    # Need gg_hat present => enable fisher_backprop for this backward
    opt.set_fisher_backprop(True)
    _run_one_backward(model)

    p0 = [p.detach().clone() for p in model.parameters()]
    k0 = int(opt._k)

    opt.step()

    assert_eq(int(opt._k), k0 + 1, "KFAC step did not increment internal counter _k")

    changed = any(not th.allclose(p.detach(), q) for p, q in zip(model.parameters(), p0))
    assert_true(changed, "KFAC step did not update parameters")


def test_kfac_conv2d_stats_shapes():
    seed_all(0)
    model = TinyCNN()
    opt = KFAC(model, lr=0.1, Ts=1, Tf=2)
    opt.set_fisher_backprop(True)

    _run_one_backward(model)

    # Find conv layer in trainable layers
    conv = None
    for m in opt._trainable_layers:
        if isinstance(m, nn.Conv2d):
            conv = m
            break
    assert_true(conv is not None, "No Conv2d layer registered")

    aa = opt._aa_hat.get(conv)
    gg = opt._gg_hat.get(conv)
    assert_true(aa is not None, "Conv2d aa_hat missing")
    assert_true(gg is not None, "Conv2d gg_hat missing")

    # aa dim: (in_ch*kh*kw + 1, same) because bias appended
    in_dim = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] + 1
    assert_eq(tuple(aa.shape), (in_dim, in_dim), "Conv2d aa_hat shape mismatch")

    # gg dim: (out_ch, out_ch)
    assert_eq(tuple(gg.shape), (conv.out_channels, conv.out_channels), "Conv2d gg_hat shape mismatch")


def test_kfac_state_dict_roundtrip():
    seed_all(0)
    model = TinyMLP()
    opt = KFAC(model, lr=0.2, Ts=1, Tf=1)
    opt.set_fisher_backprop(True)

    _run_one_backward(model)
    opt.step()

    sd = opt.state_dict()

    # Create a fresh optimizer and load
    model2 = TinyMLP()
    opt2 = KFAC(model2, lr=0.2, Ts=1, Tf=1)
    opt2.load_state_dict(sd)

    assert_eq(int(opt2._k), int(sd["k"]), "KFAC load_state_dict did not restore k")
    # Not all stats must exist, but lists should have correct length
    assert_eq(len(sd["aa_hat_list"]), len(opt2._trainable_layers), "Checkpoint list length mismatch")


def test_kfac_invalid_hparams():
    model = TinyMLP()
    assert_raises(ValueError, lambda: KFAC(model, lr=0.0))
    assert_raises(ValueError, lambda: KFAC(model, weight_decay=-1.0))
    assert_raises(ValueError, lambda: KFAC(model, damping=-1.0))
    assert_raises(ValueError, lambda: KFAC(model, momentum=1.0))
    assert_raises(ValueError, lambda: KFAC(model, eps=1.0))
    assert_raises(ValueError, lambda: KFAC(model, Ts=0))
    assert_raises(ValueError, lambda: KFAC(model, Tf=0))
    assert_raises(ValueError, lambda: KFAC(model, max_lr=0.0))
    assert_raises(ValueError, lambda: KFAC(model, trust_region=0.0))


# =============================================================================
# Tests: build_optimizer / param groups / grad clip
# =============================================================================
def test_build_optimizer_variants_and_kfac_requires_model():
    model = TinyMLP()
    params = list(model.parameters())

    o1 = build_optimizer(params, name="adamw", lr=3e-4)
    assert_true(o1.__class__.__name__.lower().startswith("adamw"), "build_optimizer adamw wrong type")

    o2 = build_optimizer(params, name="lion", lr=1e-4, lion_betas=(0.9, 0.99))
    assert_eq(o2.__class__.__name__, "Lion")

    assert_raises(ValueError, lambda: build_optimizer(params, name="kfac"))  # missing model
    o3 = build_optimizer(params, name="kfac", model=model, lr=0.2, momentum=0.9)
    assert_eq(o3.__class__.__name__, "KFAC")


def test_build_optimizer_invalid_common_args():
    model = TinyMLP()
    params = list(model.parameters())
    assert_raises(ValueError, lambda: build_optimizer(params, lr=0.0))
    assert_raises(ValueError, lambda: build_optimizer(params, weight_decay=-1.0))
    assert_raises(ValueError, lambda: build_optimizer(params, eps=0.0))
    assert_raises(ValueError, lambda: build_optimizer(params, betas=(1.0, 0.9)))


def test_make_param_groups_no_decay_and_overrides():
    model = TinyMLP()
    # make a fake "norm" param name by wrapping
    named = list(model.named_parameters())

    groups = make_param_groups(
        named,
        base_lr=1e-3,
        base_weight_decay=0.1,
        no_decay_keywords=("bias",),
        overrides=[("net.2.", {"lr": 5e-4, "weight_decay": 0.0})],  # second Linear layer prefix override
    )

    assert_true(len(groups) >= 2, "Expected at least decay and no-decay groups (plus override group possibly)")
    # bias should end up in no-decay group unless overridden by prefix
    # We just check that at least one group has weight_decay=0.0
    has_nodecay = any(float(g.get("weight_decay", -1.0)) == 0.0 for g in groups)
    assert_true(has_nodecay, "No-decay group not created")

    # override group should have lr=5e-4 if it matched any params
    has_override = any(abs(float(g.get("lr", 0.0)) - 5e-4) < 1e-12 for g in groups)
    assert_true(has_override, "Override group with lr=5e-4 not present")


def test_clip_grad_norm_basic_and_scaler_requires_optimizer():
    model = TinyMLP()
    x = th.randn(16, 8)
    y = th.randn(16, 4)
    pred = model(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()

    # max_norm <= 0 -> returns 0.0
    n0 = clip_grad_norm(model.parameters(), max_norm=0.0)
    assert_eq(n0, 0.0)

    # positive clamp -> returns float
    n1 = clip_grad_norm(model.parameters(), max_norm=0.1)
    assert_true(isinstance(n1, float) and n1 >= 0.0, "clip_grad_norm should return float >= 0")

    # scaler provided but optimizer missing -> error
    scaler = th.cuda.amp.GradScaler(enabled=False)
    assert_raises(ValueError, lambda: clip_grad_norm(model.parameters(), max_norm=1.0, scaler=scaler, optimizer=None))


# =============================================================================
# Tests: build_scheduler
# =============================================================================
def test_build_scheduler_none_and_validation():
    model = TinyMLP()
    opt = th.optim.AdamW(model.parameters(), lr=1e-3)

    s0 = build_scheduler(opt, name="none")
    assert_true(s0 is None)

    s1 = build_scheduler(opt, name="constant")
    assert_true(s1 is None)

    # progress-based schedules require total_steps
    assert_raises(ValueError, lambda: build_scheduler(opt, name="cosine", total_steps=0))
    assert_raises(ValueError, lambda: build_scheduler(opt, name="linear", total_steps=-1))
    assert_raises(ValueError, lambda: build_scheduler(opt, name="poly", total_steps=10, poly_power=0.0))

    # warmup_cosine requires warmup_steps > 0
    assert_raises(ValueError, lambda: build_scheduler(opt, name="warmup_cosine", total_steps=10, warmup_steps=0))


def test_build_scheduler_onecycle_max_lr_resolution():
    model = TinyMLP()
    opt = th.optim.AdamW(model.parameters(), lr=1e-3)

    # max_lr=None => uses current group lrs list
    s = build_scheduler(opt, name="onecycle", total_steps=20, max_lr=None)
    assert_true(s is not None)

    # max_lr sequence length mismatch => error
    assert_raises(ValueError, lambda: build_scheduler(opt, name="onecycle", total_steps=20, max_lr=[1e-3, 2e-3]))


def test_build_scheduler_step_multistep_exponential():
    model = TinyMLP()
    opt = th.optim.AdamW(model.parameters(), lr=1e-3)

    s_step = build_scheduler(opt, name="step", step_size=5, gamma=0.9)
    assert_true(s_step is not None)

    assert_raises(ValueError, lambda: build_scheduler(opt, name="step", step_size=0))
    assert_raises(ValueError, lambda: build_scheduler(opt, name="multistep", milestones=()))
    s_ms = build_scheduler(opt, name="multistep", milestones=(3, 7), gamma=0.9)
    assert_true(s_ms is not None)

    assert_raises(ValueError, lambda: build_scheduler(opt, name="exponential", gamma=0.0))
    s_exp = build_scheduler(opt, name="exponential", gamma=0.99)
    assert_true(s_exp is not None)


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    # Lion
    ("lion_step_updates_params_and_creates_state", test_lion_step_updates_params_and_creates_state),
    ("lion_invalid_hparams", test_lion_invalid_hparams),
    ("lion_sparse_grad_raises", test_lion_sparse_grad_raises),

    # KFAC
    ("kfac_registers_layers_and_hooks", test_kfac_registers_layers_and_hooks),
    ("kfac_aa_hat_updates_on_forward_pre_hook", test_kfac_aa_hat_updates_on_forward_pre_hook),
    ("kfac_gg_hat_gated_by_fisher_backprop", test_kfac_gg_hat_gated_by_fisher_backprop),
    ("kfac_step_updates_params_and_increments_k", test_kfac_step_updates_params_and_increments_k),
    ("kfac_conv2d_stats_shapes", test_kfac_conv2d_stats_shapes),
    ("kfac_state_dict_roundtrip", test_kfac_state_dict_roundtrip),
    ("kfac_invalid_hparams", test_kfac_invalid_hparams),

    # build_optimizer / utils
    ("build_optimizer_variants_and_kfac_requires_model", test_build_optimizer_variants_and_kfac_requires_model),
    ("build_optimizer_invalid_common_args", test_build_optimizer_invalid_common_args),
    ("make_param_groups_no_decay_and_overrides", test_make_param_groups_no_decay_and_overrides),
    ("clip_grad_norm_basic_and_scaler_requires_optimizer", test_clip_grad_norm_basic_and_scaler_requires_optimizer),

    # scheduler
    ("build_scheduler_none_and_validation", test_build_scheduler_none_and_validation),
    ("build_scheduler_onecycle_max_lr_resolution", test_build_scheduler_onecycle_max_lr_resolution),
    ("build_scheduler_step_multistep_exponential", test_build_scheduler_step_multistep_exponential),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="optimizers")

if __name__ == "__main__":
    raise SystemExit(main())