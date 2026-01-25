from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Tuple

import torch as th

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


from model_free.common.noises.noises import GaussianNoise, OrnsteinUhlenbeckNoise, UniformNoise
from model_free.common.noises.action_noises import (
    GaussianActionNoise,
    MultiplicativeActionNoise,
    ClippedGaussianActionNoise,
)
from model_free.common.noises.noise_builder import build_noise


# =============================================================================
# Tests: GaussianNoise
# =============================================================================
def test_gaussian_noise_shape_dtype_device_and_sigma0():
    seed_all(0)
    n = GaussianNoise(size=(3, 4), mu=0.5, sigma=0.0, device="cpu", dtype=th.float32)
    x = n.sample()
    assert_eq(tuple(x.shape), (3, 4))
    assert_eq(x.dtype, th.float32)
    assert_eq(x.device.type, "cpu")
    # sigma=0 -> constant mu
    assert_true(th.allclose(x, th.full((3, 4), 0.5)), "GaussianNoise sigma=0 should return constant mu")


def test_gaussian_noise_invalid_sigma():
    assert_raises(ValueError, lambda: GaussianNoise(size=(3,), sigma=-1.0))


def test_gaussian_noise_size_validation():
    assert_raises(ValueError, lambda: GaussianNoise(size=0))
    assert_raises(ValueError, lambda: GaussianNoise(size=()))
    assert_raises(ValueError, lambda: GaussianNoise(size=(3, -1)))


# =============================================================================
# Tests: UniformNoise
# =============================================================================
def test_uniform_noise_shape_dtype_device():
    seed_all(0)
    n = UniformNoise(size=5, low=-2.0, high=3.0, device="cpu", dtype=th.float64)
    x = n.sample()
    assert_eq(tuple(x.shape), (5,))
    assert_eq(x.dtype, th.float64)
    assert_eq(x.device.type, "cpu")
    assert_true(float(x.min()) >= -2.0 - 1e-9 and float(x.max()) <= 3.0 + 1e-9, "UniformNoise out of bounds")


def test_uniform_noise_invalid_bounds():
    assert_raises(ValueError, lambda: UniformNoise(size=3, low=1.0, high=1.0))
    assert_raises(ValueError, lambda: UniformNoise(size=3, low=2.0, high=1.0))


# =============================================================================
# Tests: OrnsteinUhlenbeckNoise
# =============================================================================
def test_ou_noise_reset_and_shape():
    seed_all(0)
    ou = OrnsteinUhlenbeckNoise(size=(4,), mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, device="cpu", dtype=th.float32)
    x0 = ou.state.clone()
    x1 = ou.sample()
    assert_eq(tuple(x1.shape), (4,))
    assert_true(not th.allclose(x0, x1), "OU sample should usually change state")
    ou.reset()
    x2 = ou.state
    assert_true(th.allclose(x2, th.zeros_like(x2)), "OU reset should restore initial state (mu when x0 None)")


def test_ou_return_copy_prevents_aliasing():
    seed_all(0)
    ou = OrnsteinUhlenbeckNoise(size=(3,), return_copy=True)
    y = ou.sample()
    # mutate returned tensor in-place; internal state should not change if clone() returned
    y.add_(1000.0)
    # next sample should be near previous internal state evolution, not huge 1000 shift
    y2 = ou.sample()
    assert_true(float(y2.abs().max()) < 100.0, "return_copy=True should prevent external aliasing corruption")


def test_ou_invalid_params():
    assert_raises(ValueError, lambda: OrnsteinUhlenbeckNoise(size=3, theta=-0.1))
    assert_raises(ValueError, lambda: OrnsteinUhlenbeckNoise(size=3, sigma=-0.1))
    assert_raises(ValueError, lambda: OrnsteinUhlenbeckNoise(size=3, dt=0.0))


# =============================================================================
# Tests: Action noises
# =============================================================================
def test_gaussian_action_noise_shape_device_dtype_and_eps_floor():
    seed_all(0)
    noise = GaussianActionNoise(sigma=0.2, eps=1e-3)
    action = th.tensor([[0.0, -2.0, 3.0]], dtype=th.float32)
    n = noise.sample(action)
    assert_eq(tuple(n.shape), tuple(action.shape))
    assert_eq(n.dtype, action.dtype)
    assert_eq(n.device, action.device)

    # First dim action=0 -> scale should be eps, so noise should not be identically zero (probabilistic)
    # We check that its absolute value is not always exactly 0.
    assert_true(float(n[0, 0].abs()) >= 0.0, "sanity")  # trivial
    assert_true(not th.allclose(n[:, 0], th.zeros_like(n[:, 0])), "eps floor should prevent zero noise at action=0")


def test_gaussian_action_noise_sigma0_returns_zero():
    noise = GaussianActionNoise(sigma=0.0)
    action = th.randn(2, 3)
    n = noise.sample(action)
    assert_true(th.allclose(n, th.zeros_like(action)), "sigma=0 should return zeros_like(action)")


def test_gaussian_action_noise_invalid_params():
    assert_raises(ValueError, lambda: GaussianActionNoise(sigma=-0.1))
    assert_raises(ValueError, lambda: GaussianActionNoise(eps=0.0))


def test_multiplicative_action_noise_sigma0_returns_zero():
    noise = MultiplicativeActionNoise(sigma=0.0)
    action = th.randn(2, 3)
    n = noise.sample(action)
    assert_true(th.allclose(n, th.zeros_like(action)), "sigma=0 should return zeros_like(action)")


def test_multiplicative_action_noise_invalid_sigma():
    assert_raises(ValueError, lambda: MultiplicativeActionNoise(sigma=-0.1))


def test_clipped_gaussian_action_noise_respects_bounds_and_returns_effective_noise():
    seed_all(0)
    noise = ClippedGaussianActionNoise(sigma=0.5, low=-1.0, high=1.0)
    action = th.tensor([[0.9, -0.9, 0.0]], dtype=th.float32)
    n = noise.sample(action)
    assert_eq(tuple(n.shape), tuple(action.shape))

    a_noisy = action + n
    assert_true(float(a_noisy.max()) <= 1.0 + 1e-6 and float(a_noisy.min()) >= -1.0 - 1e-6, "noisy action out of bounds")


def test_clipped_gaussian_action_noise_tensor_bounds_broadcast():
    seed_all(0)
    low = th.tensor([-1.0, -0.5, -2.0], dtype=th.float32)
    high = th.tensor([1.0, 0.5, 2.0], dtype=th.float32)
    noise = ClippedGaussianActionNoise(sigma=0.5, low=low, high=high)

    action = th.tensor([[0.9, -0.4, 1.5]], dtype=th.float32)
    n = noise.sample(action)
    a_noisy = action + n

    assert_true(th.all(a_noisy <= high.view(1, -1) + 1e-6).item(), "tensor high bound violated")
    assert_true(th.all(a_noisy >= low.view(1, -1) - 1e-6).item(), "tensor low bound violated")


def test_clipped_gaussian_action_noise_sigma0_returns_zero():
    noise = ClippedGaussianActionNoise(sigma=0.0, low=-1.0, high=1.0)
    action = th.randn(2, 3)
    n = noise.sample(action)
    assert_true(th.allclose(n, th.zeros_like(action)), "sigma=0 should return zeros_like(action)")


# =============================================================================
# Tests: build_noise factory
# =============================================================================
def test_build_noise_none_and_kind_normalization():
    assert_true(build_noise(kind=None, action_dim=3) is None)
    assert_true(build_noise(kind=" none ", action_dim=3) is None)
    assert_true(build_noise(kind="", action_dim=3) is None)
    assert_true(build_noise(kind="NULL", action_dim=3) is None)

    # normalization: spaces/hyphens
    n1 = build_noise(kind=" Ornstein-Uhlenbeck ", action_dim=3)
    assert_true(n1 is not None)
    assert_eq(type(n1).__name__, "OrnsteinUhlenbeckNoise")

    n2 = build_noise(kind="gaussian-action", action_dim=3)
    assert_true(n2 is not None)
    assert_eq(type(n2).__name__, "GaussianActionNoise")


def test_build_noise_invalid_action_dim_and_unknown_kind():
    assert_raises(ValueError, lambda: build_noise(kind="gaussian", action_dim=0))
    assert_raises(ValueError, lambda: build_noise(kind="weird_kind", action_dim=3))


def test_build_noise_gaussian_ou_uniform_types_and_shapes():
    g = build_noise(kind="gaussian", action_dim=5, device="cpu", dtype=th.float32)
    assert_true(g is not None)
    assert_eq(type(g).__name__, "GaussianNoise")
    x = g.sample()
    assert_eq(tuple(x.shape), (5,))
    assert_eq(x.dtype, th.float32)

    ou = build_noise(kind="ou", action_dim=2, device="cpu", dtype=th.float64)
    assert_true(ou is not None)
    assert_eq(type(ou).__name__, "OrnsteinUhlenbeckNoise")
    x2 = ou.sample()
    assert_eq(tuple(x2.shape), (2,))
    assert_eq(x2.dtype, th.float64)

    u = build_noise(kind="uniform", action_dim=4, uniform_low=-3.0, uniform_high=-1.0)
    assert_true(u is not None)
    assert_eq(type(u).__name__, "UniformNoise")
    x3 = u.sample()
    assert_eq(tuple(x3.shape), (4,))
    assert_true(float(x3.min()) >= -3.0 - 1e-6 and float(x3.max()) <= -1.0 + 1e-6)


def test_build_noise_clipped_gaussian_requires_bounds_and_validates_shapes():
    # missing bounds -> error
    assert_raises(ValueError, lambda: build_noise(kind="clipped_gaussian", action_dim=3))

    # scalar bounds ok
    n = build_noise(kind="clipped_gaussian", action_dim=3, action_noise_low=-1.0, action_noise_high=1.0)
    assert_true(n is not None)
    assert_eq(type(n).__name__, "ClippedGaussianActionNoise")

    # vector bounds length mismatch -> error
    assert_raises(
        ValueError,
        lambda: build_noise(kind="clipped_gaussian", action_dim=3, action_noise_low=[-1, -1], action_noise_high=[1, 1]),
    )

    # scalar ordering invalid -> error
    assert_raises(
        ValueError,
        lambda: build_noise(kind="clipped_gaussian", action_dim=3, action_noise_low=1.0, action_noise_high=0.0),
    )


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("gaussian_noise_shape_dtype_device_sigma0", test_gaussian_noise_shape_dtype_device_and_sigma0),
    ("gaussian_noise_invalid_sigma", test_gaussian_noise_invalid_sigma),
    ("gaussian_noise_size_validation", test_gaussian_noise_size_validation),

    ("uniform_noise_shape_dtype_device", test_uniform_noise_shape_dtype_device),
    ("uniform_noise_invalid_bounds", test_uniform_noise_invalid_bounds),

    ("ou_noise_reset_and_shape", test_ou_noise_reset_and_shape),
    ("ou_return_copy_prevents_aliasing", test_ou_return_copy_prevents_aliasing),
    ("ou_invalid_params", test_ou_invalid_params),

    ("gaussian_action_noise_shape_device_dtype_eps_floor", test_gaussian_action_noise_shape_device_dtype_and_eps_floor),
    ("gaussian_action_noise_sigma0_returns_zero", test_gaussian_action_noise_sigma0_returns_zero),
    ("gaussian_action_noise_invalid_params", test_gaussian_action_noise_invalid_params),

    ("multiplicative_action_noise_sigma0_returns_zero", test_multiplicative_action_noise_sigma0_returns_zero),
    ("multiplicative_action_noise_invalid_sigma", test_multiplicative_action_noise_invalid_sigma),

    ("clipped_gaussian_action_noise_respects_bounds", test_clipped_gaussian_action_noise_respects_bounds_and_returns_effective_noise),
    ("clipped_gaussian_action_noise_tensor_bounds_broadcast", test_clipped_gaussian_action_noise_tensor_bounds_broadcast),
    ("clipped_gaussian_action_noise_sigma0_returns_zero", test_clipped_gaussian_action_noise_sigma0_returns_zero),

    ("build_noise_none_and_normalization", test_build_noise_none_and_kind_normalization),
    ("build_noise_invalid_action_dim_and_unknown_kind", test_build_noise_invalid_action_dim_and_unknown_kind),
    ("build_noise_gaussian_ou_uniform_types_shapes", test_build_noise_gaussian_ou_uniform_types_and_shapes),
    ("build_noise_clipped_gaussian_requires_bounds_and_validates_shapes", test_build_noise_clipped_gaussian_requires_bounds_and_validates_shapes),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="noises")

if __name__ == "__main__":
    raise SystemExit(main())