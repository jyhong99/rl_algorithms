from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Tuple

import numpy as np
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

from model_free.common.networks.base_networks import (
    MLPFeaturesExtractor,
    NoisyLinear,
    NoisyMLPFeaturesExtractor,
)
from model_free.common.networks.policy_networks import (
    DeterministicPolicyNetwork,
    ContinuousPolicyNetwork,
    DiscretePolicyNetwork,
)
from model_free.common.networks.q_networks import (
    QNetwork,
    DoubleQNetwork,
    QuantileQNetwork,
    RainbowQNetwork,
)
from model_free.common.networks.value_networks import (
    StateValueNetwork,
    StateActionValueNetwork,
    DoubleStateActionValueNetwork,
    QuantileStateActionValueNetwork,
)

from model_free.common.networks.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    CategoricalDistribution,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


# =============================================================================
# Tests: MLPFeaturesExtractor
# =============================================================================
def test_mlp_features_extractor_shape_and_out_dim():
    net = MLPFeaturesExtractor(input_dim=5, hidden_sizes=[16, 8])
    x = th.randn(4, 5)
    y = net(x)
    assert_eq(tuple(y.shape), (4, 8))
    assert_eq(net.out_dim, 8)


def test_mlp_features_extractor_requires_hidden_sizes():
    assert_raises(ValueError, lambda: MLPFeaturesExtractor(input_dim=5, hidden_sizes=[]))


# =============================================================================
# Tests: NoisyLinear / NoisyMLPFeaturesExtractor
# =============================================================================
def test_noisylinear_reset_noise_changes_output():
    seed_all(0)
    layer = NoisyLinear(4, 3)
    x = th.randn(2, 4)

    y1 = layer(x)
    layer.reset_noise()
    y2 = layer(x)

    # With high probability, outputs differ after reset_noise.
    assert_true(not th.allclose(y1, y2), "NoisyLinear output did not change after reset_noise()")


def test_noisymlp_reset_noise_changes_output():
    seed_all(0)
    net = NoisyMLPFeaturesExtractor(input_dim=4, hidden_sizes=(8, 8))
    x = th.randn(2, 4)
    y1 = net(x)
    net.reset_noise()
    y2 = net(x)
    assert_true(not th.allclose(y1, y2), "NoisyMLPFeaturesExtractor output did not change after reset_noise()")


# =============================================================================
# Tests: Distributions
# =============================================================================
def test_diag_gaussian_shapes_logp_entropy_mode():
    mean = th.zeros(5, 3)
    log_std = th.zeros(5, 3)
    dist = DiagGaussianDistribution(mean, log_std)

    a = dist.sample()
    assert_eq(tuple(a.shape), (5, 3))

    lp = dist.log_prob(a)
    ent = dist.entropy()
    m = dist.mode()
    assert_eq(tuple(lp.shape), (5, 1))
    assert_eq(tuple(ent.shape), (5, 1))
    assert_eq(tuple(m.shape), (5, 3))

    # log_std clamp bounds (sanity)
    assert_true(float(dist.log_std.min()) >= LOG_STD_MIN - 1e-6)
    assert_true(float(dist.log_std.max()) <= LOG_STD_MAX + 1e-6)


def test_squashed_gaussian_logp_pre_tanh_path_matches_inverse_path():
    seed_all(0)
    mean = th.zeros(4, 2)
    log_std = th.zeros(4, 2)
    dist = SquashedDiagGaussianDistribution(mean, log_std)

    a, z = dist.rsample(return_pre_tanh=True)
    lp1 = dist.log_prob(a, pre_tanh=z)
    lp2 = dist.log_prob(a, pre_tanh=None)

    assert_eq(tuple(lp1.shape), (4, 1))
    assert_eq(tuple(lp2.shape), (4, 1))
    # should be close (inverse has numerical clamp, so allow small tolerance)
    assert_true(th.allclose(lp1, lp2, atol=1e-5, rtol=1e-5), "log_prob mismatch between pre_tanh vs inverse path")


def test_categorical_distribution_shapes():
    logits = th.randn(6, 5)
    dist = CategoricalDistribution(logits)

    a = dist.sample()
    assert_eq(tuple(a.shape), (6, 1))
    assert_true(a.dtype in (th.int64, th.long))

    lp = dist.log_prob(a)
    ent = dist.entropy()
    m = dist.mode()
    assert_eq(tuple(lp.shape), (6, 1))
    assert_eq(tuple(ent.shape), (6, 1))
    assert_eq(tuple(m.shape), (6, 1))


# =============================================================================
# Tests: Policy networks
# =============================================================================
def test_deterministic_policy_action_shape_and_bounds_scaling():
    seed_all(0)
    obs_dim, act_dim = 7, 3

    low = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    high = np.array([2.0, 1.0, 4.0], dtype=np.float32)

    pi = DeterministicPolicyNetwork(obs_dim, act_dim, hidden_sizes=[32, 32], action_low=low, action_high=high)
    obs = th.randn(5, obs_dim)
    a = pi(obs)
    assert_eq(tuple(a.shape), (5, act_dim))

    # must be within [low, high] (because tanh -> [-1,1] then affine map)
    a_np = a.detach().cpu().numpy()
    assert_true(np.all(a_np >= low - 1e-5) and np.all(a_np <= high + 1e-5), "scaled action out of bounds")


def test_deterministic_policy_act_noise_and_clip():
    seed_all(0)
    obs_dim, act_dim = 5, 2
    low = np.array([-1.0, -1.0], np.float32)
    high = np.array([1.0, 1.0], np.float32)
    pi = DeterministicPolicyNetwork(obs_dim, act_dim, hidden_sizes=[16], action_low=low, action_high=high)

    obs = th.randn(obs_dim)  # (obs_dim,) should be auto-batched
    a_det, info_det = pi.act(obs, deterministic=True, noise_std=0.5, clip=True)
    a_sto, info_sto = pi.act(obs, deterministic=False, noise_std=0.5, clip=True)

    assert_eq(tuple(a_det.shape), (1, act_dim))
    assert_eq(tuple(a_sto.shape), (1, act_dim))
    assert_eq(tuple(info_det["noise"].shape), (1, act_dim))
    assert_eq(tuple(info_sto["noise"].shape), (1, act_dim))

    # deterministic=True -> noise should be zeros
    assert_true(th.allclose(info_det["noise"], th.zeros_like(info_det["noise"])), "deterministic act produced noise")
    # stochastic -> noise likely nonzero
    assert_true(not th.allclose(info_sto["noise"], th.zeros_like(info_sto["noise"])), "stochastic act noise is zero")

    # clip True and bounds exist => action in bounds
    assert_true(float(a_sto.max()) <= 1.0 + 1e-5 and float(a_sto.min()) >= -1.0 - 1e-5)


def test_continuous_policy_act_logp_shapes_unsquashed():
    seed_all(0)
    obs_dim, act_dim = 6, 4
    pi = ContinuousPolicyNetwork(obs_dim, act_dim, hidden_sizes=[32, 32], squash=False, log_std_mode="param")

    obs = th.randn(3, obs_dim)
    a, info = pi.act(obs, deterministic=False, return_logp=True)
    assert_eq(tuple(a.shape), (3, act_dim))
    assert_true("logp" in info)
    assert_eq(tuple(info["logp"].shape), (3, 1))


def test_continuous_policy_act_logp_shapes_squashed():
    seed_all(0)
    obs_dim, act_dim = 6, 4
    pi = ContinuousPolicyNetwork(obs_dim, act_dim, hidden_sizes=[32, 32], squash=True, log_std_mode="layer")

    obs = th.randn(3, obs_dim)
    a, info = pi.act(obs, deterministic=False, return_logp=True)
    assert_eq(tuple(a.shape), (3, act_dim))
    assert_true("logp" in info)
    assert_eq(tuple(info["logp"].shape), (3, 1))
    # squashed actions should be in [-1,1]
    assert_true(float(a.max()) <= 1.0 + 1e-5 and float(a.min()) >= -1.0 - 1e-5)


def test_discrete_policy_act_shapes_and_logp():
    seed_all(0)
    obs_dim, n_actions = 8, 7
    pi = DiscretePolicyNetwork(obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=[32, 32])

    obs = th.randn(4, obs_dim)
    a, info = pi.act(obs, deterministic=False, return_logp=True)
    assert_eq(tuple(a.shape), (4, 1))
    assert_eq(tuple(info["logp"].shape), (4, 1))
    # action indices must be in [0, n_actions-1]
    assert_true(int(a.min()) >= 0 and int(a.max()) < n_actions)


# =============================================================================
# Tests: Value / critic networks
# =============================================================================
def test_qnetwork_forward_shapes_dueling_and_non_dueling():
    seed_all(0)
    state_dim, action_dim = 10, 5
    x = th.randn(6, state_dim)

    q1 = QNetwork(state_dim=state_dim, action_dim=action_dim, dueling_mode=False)
    y1 = q1(x)
    assert_eq(tuple(y1.shape), (6, action_dim))

    q2 = QNetwork(state_dim=state_dim, action_dim=action_dim, dueling_mode=True)
    y2 = q2(x)
    assert_eq(tuple(y2.shape), (6, action_dim))


def test_double_qnetwork_shapes():
    seed_all(0)
    net = DoubleQNetwork(state_dim=9, action_dim=4, dueling=False)
    x = th.randn(3, 9)
    q1, q2 = net(x)
    assert_eq(tuple(q1.shape), (3, 4))
    assert_eq(tuple(q2.shape), (3, 4))


def test_quantile_qnetwork_shape():
    seed_all(0)
    state_dim, action_dim, n_quantiles = 8, 3, 25
    net = QuantileQNetwork(state_dim=state_dim, action_dim=action_dim, n_quantiles=n_quantiles, dueling_mode=False)
    x = th.randn(4, state_dim)
    out = net(x)
    assert_eq(tuple(out.shape), (4, n_quantiles, action_dim))


def test_quantile_qnetwork_dueling_shape():
    seed_all(0)
    state_dim, action_dim, n_quantiles = 8, 3, 25
    net = QuantileQNetwork(state_dim=state_dim, action_dim=action_dim, n_quantiles=n_quantiles, dueling_mode=True)
    x = th.randn(4, state_dim)
    out = net(x)
    assert_eq(tuple(out.shape), (4, n_quantiles, action_dim))


def test_rainbow_qnetwork_dist_normalized_and_forward_shape():
    seed_all(0)
    state_dim, action_dim, atom_size = 7, 4, 51
    support = th.linspace(-10.0, 10.0, atom_size)
    net = RainbowQNetwork(state_dim=state_dim, action_dim=action_dim, atom_size=atom_size, support=support)

    x = th.randn(5, state_dim)
    dist = net.dist(x)
    assert_eq(tuple(dist.shape), (5, action_dim, atom_size))
    # sums to 1 across atoms
    s = dist.sum(dim=-1)
    assert_true(th.allclose(s, th.ones_like(s), atol=1e-5, rtol=1e-5), "Rainbow dist not normalized")

    q = net(x)
    assert_eq(tuple(q.shape), (5, action_dim))

    # noise reset should change distribution (high probability)
    dist1 = net.dist(x)
    net.reset_noise()
    dist2 = net.dist(x)
    assert_true(not th.allclose(dist1, dist2), "Rainbow dist did not change after reset_noise()")


def test_state_value_network_shape():
    seed_all(0)
    net = StateValueNetwork(state_dim=6)
    x = th.randn(4, 6)
    v = net(x)
    assert_eq(tuple(v.shape), (4, 1))


def test_state_action_value_network_shape():
    seed_all(0)
    net = StateActionValueNetwork(state_dim=6, action_dim=2)
    s = th.randn(4, 6)
    a = th.randn(4, 2)
    q = net(s, a)
    assert_eq(tuple(q.shape), (4, 1))


def test_double_state_action_value_network_shapes():
    seed_all(0)
    net = DoubleStateActionValueNetwork(state_dim=6, action_dim=2)
    s = th.randn(4, 6)
    a = th.randn(4, 2)
    q1, q2 = net(s, a)
    assert_eq(tuple(q1.shape), (4, 1))
    assert_eq(tuple(q2.shape), (4, 1))


def test_quantile_state_action_value_network_shape():
    seed_all(0)
    net = QuantileStateActionValueNetwork(state_dim=5, action_dim=3, n_quantiles=25, n_nets=2)
    s = th.randn(7, 5)
    a = th.randn(7, 3)
    out = net(s, a)
    assert_eq(tuple(out.shape), (7, 2, 25))


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    # base trunks
    ("mlp_features_shape", test_mlp_features_extractor_shape_and_out_dim),
    ("mlp_features_requires_hidden", test_mlp_features_extractor_requires_hidden_sizes),

    # noisy
    ("noisylinear_reset_noise_changes_output", test_noisylinear_reset_noise_changes_output),
    ("noisymlp_reset_noise_changes_output", test_noisymlp_reset_noise_changes_output),

    # distributions
    ("diag_gaussian_shapes", test_diag_gaussian_shapes_logp_entropy_mode),
    ("squashed_gaussian_logp_pre_tanh_matches_inverse", test_squashed_gaussian_logp_pre_tanh_path_matches_inverse_path),
    ("categorical_shapes", test_categorical_distribution_shapes),

    # policies
    ("det_policy_bounds_scaling", test_deterministic_policy_action_shape_and_bounds_scaling),
    ("det_policy_act_noise_clip", test_deterministic_policy_act_noise_and_clip),
    ("cont_policy_unsquashed_act_logp", test_continuous_policy_act_logp_shapes_unsquashed),
    ("cont_policy_squashed_act_logp", test_continuous_policy_act_logp_shapes_squashed),
    ("disc_policy_act_logp", test_discrete_policy_act_shapes_and_logp),

    # value/critic
    ("qnetwork_shapes", test_qnetwork_forward_shapes_dueling_and_non_dueling),
    ("double_qnetwork_shapes", test_double_qnetwork_shapes),
    ("quantile_qnetwork_shape", test_quantile_qnetwork_shape),
    ("quantile_qnetwork_dueling_shape", test_quantile_qnetwork_dueling_shape),
    ("rainbow_dist_normalized_and_noise_reset", test_rainbow_qnetwork_dist_normalized_and_forward_shape),
    ("state_value_shape", test_state_value_network_shape),
    ("state_action_value_shape", test_state_action_value_network_shape),
    ("double_state_action_value_shapes", test_double_state_action_value_network_shapes),
    ("quantile_state_action_value_shape", test_quantile_state_action_value_network_shape),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="networks")

if __name__ == "__main__":
    raise SystemExit(main())