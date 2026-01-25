from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


def _bootstrap_sys_path() -> None:
    """
    Make `model_free` importable when running this tester as a standalone script.

    Strategy
    --------
    Walk upward a few directories; if a parent contains "model_free", prepend it to sys.path.
    Fallback to a relative parent-of-parent insertion.
    """
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

# IMPORTANT:
# Adjust this import path to your actual discrete PPO builder location.
# The tester assumes you have a builder function named `ppo` that builds discrete PPO.
from model_free.baselines.policy_gradients.on_policy.ppo_discrete.ppo_discrete import ppo_discrete  # noqa: E402
from model_free.baselines.policy_gradients.on_policy.ppo_discrete.head import PPODiscreteHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
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
    Tiny episodic env mimicking CartPole shapes (discrete actions):

    - obs: (4,) float32
    - actions: {0,1}
    - episode ends after horizon steps
    - reward: 1.0 each step
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
            self.__post_init__()  # re-init rng/spaces deterministically
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
    """Flatten all parameters into a single float32 CPU vector for change-detection tests."""
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(*, device: str = "cpu"):
    """
    Build a small discrete PPO algorithm instance for smoke tests.

    IMPORTANT:
    - obs_dim=4, action_dim=2 is consistent with DummyCartPoleLikeEnv.
    - rollout_steps is small to make tests run quickly.
    """
    return ppo_discrete(
        obs_dim=4,
        n_actions=2,
        device=device,
        hidden_sizes=(64, 64),
        rollout_steps=128,
        update_epochs=2,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        use_amp=False,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Drive env for n_steps, pushing transitions into algo and optionally running updates.

    This uses the *on-policy* contract:
      - algo.act(obs) -> action index
      - algo.on_env_step(transition_dict)
      - algo.ready_to_update() / algo.update()

    Transition dict fields:
      obs, action, reward, next_obs, done

    NOTE:
    PPO generally needs old_logp/value stored at collection-time. Your OnPolicyAlgorithm
    likely computes and stores them inside on_env_step via head.evaluate_actions(...).
    That requires action shape consistency; we provide action as np.int64 (scalar), but
    we also provide a safe (1,) form via np.asarray when needed.
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)
        a_int = int(a)

        next_obs, r, terminated, truncated, _info = env.step(a_int)
        done = bool(terminated or truncated)

        # IMPORTANT:
        # Some discrete heads enforce action tensor shape (B,) or (B,1) and reject scalar ().
        # To be robust across implementations, pass an explicit (1,) array.
        act_payload = np.asarray([a_int], dtype=np.int64)

        algo.on_env_step(
            {
                "obs": obs,
                "action": act_payload,
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
            }
        )

        if do_updates and algo.ready_to_update():
            last_metrics = algo.update()
            num_updates += 1

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return {"num_updates": int(num_updates), "last_metrics": last_metrics}


# =============================================================================
# Tests: PPODiscreteHead
# =============================================================================
def test_ppo_discrete_head_act_and_evaluate_shapes():
    seed_all(0)

    head = PPODiscreteHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # act() should return a tensor of action indices with batch dimension
    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_true(tuple(a.shape) == (1,) or tuple(a.shape) == (1, 1), f"unexpected act shape: {tuple(a.shape)}")

    # evaluate_actions() should return value/log_prob/entropy with (B,1)
    # Provide action in a tolerant form (B,) long.
    act_idx = th.tensor([0], dtype=th.int64)
    out = head.evaluate_actions(obs, act_idx)

    assert_true(isinstance(out, dict), "evaluate_actions must return dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing required keys")
    assert_true(th.is_tensor(out["value"]) and th.is_tensor(out["log_prob"]) and th.is_tensor(out["entropy"]))
    assert_eq(tuple(out["value"].shape), (1, 1))
    assert_eq(tuple(out["log_prob"].shape), (1, 1))
    assert_eq(tuple(out["entropy"].shape), (1, 1))


def test_ppo_discrete_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = PPODiscreteHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ppo_discrete_head_ckpt")
        head1.save(path)

        head2 = PPODiscreteHead(
            obs_dim=4,
            n_actions=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
        )
        head2.load(path)

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
# Tests: OnPolicyAlgorithm integration (Discrete PPO)
# =============================================================================
def test_ppo_discrete_algorithm_smoke_run_runs_and_updates():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu")
    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # These keys depend on your PPO core logging; keep them broad but meaningful.
    # Adjust if your metrics namespaces differ.
    must_keys = [
        "loss/policy",
        "loss/value",
        "loss/total",
        "stats/entropy",
        "lr/actor",
        "lr/critic",
    ]
    for k in must_keys:
        assert_true(k in m, f"missing metric key: {k}")

    assert_true(np.isfinite(float(m["loss/policy"])), "loss/policy not finite")
    assert_true(np.isfinite(float(m["loss/value"])), "loss/value not finite")
    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")


def test_ppo_discrete_parameters_change_after_update():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu")
    algo.setup(env)

    # Drive until ready_to_update
    obs, _ = env.reset(seed=0)
    drove = 0
    max_drive = 5000
    while drove < max_drive and (not algo.ready_to_update()):
        a = int(algo.act(obs, deterministic=False))
        next_obs, r, terminated, truncated, _ = env.step(a)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": np.asarray([a], dtype=np.int64),
                "reward": float(r),
                "next_obs": next_obs,
                "done": done,
            }
        )
        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    actor_before = _state_vector(algo.head.actor)
    critic_before = _state_vector(algo.head.critic)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    actor_after = _state_vector(algo.head.actor)
    critic_after = _state_vector(algo.head.critic)

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    dc = float(th.linalg.vector_norm(critic_after - critic_before).item()) if critic_before.numel() else 0.0

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")
    env.close()


def test_ppo_discrete_action_shape_is_tolerant_in_algorithm_path():
    """
    Regression test:
    Some heads enforce discrete action tensors to be shaped (B,) or (B,1) and will
    reject scalar shape ().

    This test verifies our tester/runner passes a (1,) action array, and that
    OnPolicyAlgorithm accepts it.
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=10, seed=0)

    algo = _make_small_algo(device="cpu")
    algo.setup(env)

    obs, _ = env.reset(seed=0)
    a = int(env.action_space.sample())
    next_obs, r, terminated, truncated, _ = env.step(a)
    done = bool(terminated or truncated)

    # Provide (1,) instead of scalar.
    algo.on_env_step(
        {
            "obs": obs,
            "action": np.asarray([a], dtype=np.int64),
            "reward": float(r),
            "next_obs": next_obs,
            "done": done,
        }
    )
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("ppo_discrete_head_act_and_evaluate_shapes", test_ppo_discrete_head_act_and_evaluate_shapes),
    ("ppo_discrete_head_save_and_load_roundtrip", test_ppo_discrete_head_save_and_load_roundtrip),
    ("ppo_discrete_algorithm_smoke_run_runs_and_updates", test_ppo_discrete_algorithm_smoke_run_runs_and_updates),
    ("ppo_discrete_parameters_change_after_update", test_ppo_discrete_parameters_change_after_update),
    ("ppo_discrete_action_shape_is_tolerant_in_algorithm_path", test_ppo_discrete_action_shape_is_tolerant_in_algorithm_path),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="ppo_discrete")


if __name__ == "__main__":
    raise SystemExit(main())
