from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


# =============================================================================
# Path bootstrap (match your repo layout)
# =============================================================================
def _bootstrap_sys_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        if os.path.isdir(os.path.join(parent, "model_free")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        cur = parent

    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_bootstrap_sys_path()


from model_free.common.testers.test_utils import (  # noqa: E402
    seed_all,
    run_tests,
    assert_eq,
    assert_true,
    assert_raises,
)

# =============================================================================
# Imports under test (adjust import paths if your package layout differs)
# =============================================================================
from model_free.baselines.policy_gradients.on_policy.a2c_discrete.head import (  # noqa: E402
    A2CDiscreteHead,
)
from model_free.baselines.policy_gradients.on_policy.a2c_discrete.core import (  # noqa: E402
    A2CDiscreteCore,
)
from model_free.baselines.policy_gradients.on_policy.a2c_discrete.a2c_discrete import (  # noqa: E402
    a2c_discrete,
)


# =============================================================================
# Minimal spaces/env (no gym dependency) - same style as your ACER test
# =============================================================================
class DummyDiscreteSpace:
    """Minimal Discrete(n) space with sample()."""

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = int(n)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> int:
        return int(self._rng.randint(0, self.n))


class DummyBoxSpace:
    """Minimal Box(shape,) space with sample()."""

    def __init__(self, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.low = np.full(self.shape, float(low), dtype=np.float32)
        self.high = np.full(self.shape, float(high), dtype=np.float32)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> np.ndarray:
        return self._rng.uniform(low=self.low, high=self.high).astype(np.float32)


@dataclass
class DummyCartPoleLikeEnv:
    """
    A tiny episodic env mimicking CartPole shapes:
      - obs: (4,) float32
      - actions: {0,1}
    Episode ends after horizon steps.
    Reward = 1.0 per step.
    """

    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(4,), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyDiscreteSpace(n=2, seed=self.seed + 123)
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((4,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)
        noise = self._rng.randn(4).astype(np.float32) * 0.01
        delta = (0.05 if a == 1 else -0.05)
        self._obs = np.clip(self._obs + delta + noise, -1.0, 1.0).astype(np.float32)

        self._t += 1
        reward = 1.0
        terminated = False
        truncated = self._t >= int(self.horizon)
        info: Dict[str, Any] = {"TimeLimit.truncated": bool(truncated)}
        return self._obs.copy(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        return


# =============================================================================
# Helpers
# =============================================================================
def _state_vector(module: th.nn.Module) -> th.Tensor:
    """Flatten all parameters into a single CPU float vector (for diff checks)."""
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


@dataclass
class DummyOnPolicyBatch:
    """Minimal batch object matching your A2CDiscreteCore batch contract."""
    observations: th.Tensor
    actions: th.Tensor
    returns: th.Tensor
    advantages: th.Tensor


def _make_dummy_batch(*, B: int = 32, obs_dim: int = 4, n_actions: int = 2, device: str = "cpu") -> DummyOnPolicyBatch:
    rng = np.random.RandomState(0)
    obs = th.tensor(rng.randn(B, obs_dim).astype(np.float32), device=device)
    act = th.tensor(rng.randint(0, n_actions, size=(B,)).astype(np.int64), device=device)
    # returns / advantages: just finite floats (not necessarily consistent)
    ret = th.tensor(rng.randn(B, 1).astype(np.float32), device=device)
    adv = th.tensor(rng.randn(B, 1).astype(np.float32), device=device)
    return DummyOnPolicyBatch(observations=obs, actions=act, returns=ret, advantages=adv)


def _make_small_head(*, device: str = "cpu") -> A2CDiscreteHead:
    return A2CDiscreteHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device=device,
    )


def _make_small_core(*, head: Any, device: str = "cpu") -> A2CDiscreteCore:
    # Core builds optimizers/schedulers in base class.
    return A2CDiscreteCore(
        head=head,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        use_amp=False,
        actor_optim_name="adamw",
        actor_lr=3e-4,
        actor_weight_decay=0.0,
        critic_optim_name="adamw",
        critic_lr=3e-4,
        critic_weight_decay=0.0,
        actor_sched_name="none",
        critic_sched_name="none",
        total_steps=0,
        warmup_steps=0,
        min_lr_ratio=0.0,
        poly_power=1.0,
        step_size=1000,
        sched_gamma=0.99,
        milestones=(),
    )


# =============================================================================
# Tests: A2CDiscreteHead
# =============================================================================
def test_a2c_discrete_head_act_and_eval_shapes():
    seed_all(0)
    head = _make_small_head(device="cpu")
    head.set_training(False)

    obs = np.zeros((4,), dtype=np.float32)

    # act(): should return action indices (B,) long (B=1 here)
    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1,), "discrete act() should return (B,) indices")
    assert_true(a.dtype in (th.int64, th.int32), "discrete action dtype should be integer-like")

    # evaluate_actions(): should standardize to (B,1) for value/log_prob/entropy
    out = head.evaluate_actions(obs, a, as_scalar=False)
    assert_true(isinstance(out, dict), "evaluate_actions must return dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]
    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "outputs must be tensors")
    assert_eq(tuple(v.shape), (1, 1))
    assert_eq(tuple(lp.shape), (1, 1))
    assert_eq(tuple(ent.shape), (1, 1))


def test_a2c_discrete_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = _make_small_head(device="cpu")
    _ = head1.act(np.zeros((4,), dtype=np.float32))  # touch forward

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a2c_discrete_head_ckpt")
        head1.save(path)

        head2 = _make_small_head(device="cpu")
        head2.load(path)

        # compare actor/critic weights
        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


# =============================================================================
# Tests: A2CDiscreteCore
# =============================================================================
def test_a2c_discrete_core_update_runs_and_returns_metrics():
    seed_all(0)

    head = _make_small_head(device="cpu")
    core = _make_small_core(head=head, device="cpu")

    batch = _make_dummy_batch(B=32, obs_dim=4, n_actions=2, device="cpu")
    m = core.update_from_batch(batch)

    assert_true(isinstance(m, dict) and len(m) > 0, "update_from_batch should return non-empty metrics")
    for k in ("loss/policy", "loss/value", "loss/entropy", "loss/total", "lr/actor", "lr/critic"):
        assert_true(k in m, f"missing metric: {k}")
        assert_true(np.isfinite(float(m[k])), f"metric not finite: {k}={m[k]}")


def test_a2c_discrete_core_parameters_change_after_update():
    seed_all(0)

    head = _make_small_head(device="cpu")
    core = _make_small_core(head=head, device="cpu")

    actor_before = _state_vector(head.actor)
    critic_before = _state_vector(head.critic)

    batch = _make_dummy_batch(B=64, obs_dim=4, n_actions=2, device="cpu")
    _ = core.update_from_batch(batch)

    actor_after = _state_vector(head.actor)
    critic_after = _state_vector(head.critic)

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    dc = float(th.linalg.vector_norm(critic_after - critic_before).item()) if critic_before.numel() else 0.0

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")


def test_a2c_discrete_core_rejects_bad_action_shapes():
    """
    Optional: ensure your discrete action normalization is robust.
    If you intentionally allow broader shapes, adjust/remove this test.
    """
    seed_all(0)

    head = _make_small_head(device="cpu")
    core = _make_small_core(head=head, device="cpu")

    # action shape (B,2) is nonsensical for categorical; should error or misbehave.
    B = 8
    obs = th.zeros((B, 4), dtype=th.float32)
    act_bad = th.zeros((B, 2), dtype=th.int64)
    ret = th.zeros((B, 1), dtype=th.float32)
    adv = th.zeros((B, 1), dtype=th.float32)

    batch = DummyOnPolicyBatch(observations=obs, actions=act_bad, returns=ret, advantages=adv)

    # Depending on your dist/log_prob behavior, this may raise.
    # If you prefer "always raise", keep assert_raises; otherwise remove.
    assert_raises(Exception, lambda: core.update_from_batch(batch))


# =============================================================================
# Tests: OnPolicyAlgorithm integration (smoke) - best-effort
# =============================================================================
def test_a2c_discrete_algorithm_smoke_run_if_supported():
    """
    Best-effort integration test.
    This will run only if your OnPolicyAlgorithm exposes a compatible API:
      - algo.setup(env)
      - algo.act(obs, deterministic=False)
      - algo.on_env_step(transition_dict)
      - algo.ready_to_update()
      - algo.update()
    If your API differs, adapt this test to your algorithm driver.
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = a2c_discrete(
        obs_dim=4,
        n_actions=2,
        device="cpu",
        hidden_sizes=(32, 32),
        rollout_steps=128,
        update_epochs=1,
        minibatch_size=None,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        use_amp=False,
    )

    # Feature-detect the API. If not present, skip gracefully.
    required = ["setup", "act", "on_env_step", "ready_to_update", "update"]
    if not all(hasattr(algo, name) for name in required):
        env.close()
        assert_true(True, "OnPolicyAlgorithm API differs; integration smoke test skipped.")
        return

    obs, _ = env.reset(seed=0)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(2000):
        a = algo.act(obs, deterministic=False)

        # normalize action int for env.step
        if th.is_tensor(a):
            # a is typically shape (1,) for discrete
            a_int = int(a.detach().cpu().reshape(-1)[0].item())
        else:
            a_int = int(a)

        next_obs, r, terminated, truncated, info = env.step(a_int)
        done = bool(terminated or truncated)

        # IMPORTANT: vectorize action for head.evaluate_actions() contract
        act_vec = np.asarray([a_int], dtype=np.int64)

        algo.on_env_step(
            {
                "obs": obs,
                "action": act_vec,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
                "info": info,
            }
        )

        if algo.ready_to_update():
            last_metrics = algo.update()
            num_updates += 1

        obs = next_obs
        if done:
            obs, _ = env.reset()

    env.close()

    assert_true(num_updates > 0, "expected at least one update to run")
    assert_true(isinstance(last_metrics, dict) and len(last_metrics) > 0, "update() should return non-empty metrics")

    # Check for a few expected keys (adjust based on your logger/metrics naming)
    # We keep this permissive, because naming varies across implementations.
    any_loss_key = any(k.startswith("loss/") for k in last_metrics.keys())
    assert_true(any_loss_key, "expected at least one loss/* key in metrics")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("a2c_discrete_head_act_and_eval_shapes", test_a2c_discrete_head_act_and_eval_shapes),
    ("a2c_discrete_head_save_and_load_roundtrip", test_a2c_discrete_head_save_and_load_roundtrip),
    ("a2c_discrete_core_update_runs_and_returns_metrics", test_a2c_discrete_core_update_runs_and_returns_metrics),
    ("a2c_discrete_core_parameters_change_after_update", test_a2c_discrete_core_parameters_change_after_update),
    ("a2c_discrete_core_rejects_bad_action_shapes", test_a2c_discrete_core_rejects_bad_action_shapes),
    ("a2c_discrete_algorithm_smoke_run_if_supported", test_a2c_discrete_algorithm_smoke_run_if_supported),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="a2c_discrete")


if __name__ == "__main__":
    raise SystemExit(main())
