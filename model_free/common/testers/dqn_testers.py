from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


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

# NOTE: adjust to your actual module path
from model_free.baselines.q_learning.dqn.dqn import dqn  # noqa: E402
from model_free.baselines.q_learning.dqn.head import DQNHead  # noqa: E402


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
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(*, device: str = "cpu", use_per: bool = True):
    """
    Small DQN instance for fast smoke tests.

    IMPORTANT
    ---------
    We rely on OffPolicyAlgorithm's new epsilon scheduling.
    So we only call algo.act(obs, deterministic=False) in rollouts.
    """
    return dqn(
        obs_dim=4,
        n_actions=2,
        device=device,
        hidden_sizes=(64, 64),
        # replay/schedule
        buffer_size=10_000,
        batch_size=32,
        warmup_env_steps=50,
        update_after=50,
        update_every=1,
        utd=1.0,
        gradient_steps=1,
        max_updates_per_call=10,
        # PER
        use_per=bool(use_per),
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_final=0.8,
        per_beta_anneal_steps=1_000,
        # core
        gamma=0.99,
        double_dqn=True,
        huber=True,
        target_update_interval=50,
        tau=0.0,
        max_grad_norm=10.0,
        use_amp=False,
        # exploration (NEW)
        exploration_eps=0.2,
        exploration_eps_final=0.05,
        exploration_eps_anneal_steps=500,
        exploration_eval_eps=0.0,
    )


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        # OffPolicyAlgorithm.act handles:
        #  - warmup random action
        #  - epsilon-greedy (if head.act supports epsilon)
        a = algo.act(obs, deterministic=False)
        a_int = int(np.asarray(a).reshape(-1)[0]) if isinstance(a, (np.ndarray, list, tuple)) else int(a)

        next_obs, r, terminated, truncated, _info = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_int,
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
# Tests: DQNHead
# =============================================================================
def test_dqn_head_q_values_and_act_shapes():
    seed_all(0)

    head = DQNHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=False,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    q = head.q_values(obs)
    assert_true(th.is_tensor(q), "head.q_values should return torch.Tensor")
    assert_eq(tuple(q.shape), (1, 2))

    qt = head.q_values_target(obs)
    assert_true(th.is_tensor(qt), "head.q_values_target should return torch.Tensor")
    assert_eq(tuple(qt.shape), (1, 2))

    # DQNHead.act should accept epsilon (head-level API).
    a = head.act(obs, epsilon=0.0, deterministic=True)
    a0 = int(np.asarray(a).reshape(-1)[0]) if isinstance(a, (np.ndarray, list, tuple)) else int(a)
    assert_true(a0 in (0, 1), f"action out of range: {a0}")


def test_dqn_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = DQNHead(
        obs_dim=4,
        n_actions=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        dueling_mode=True,
        device="cpu",
    )

    _ = head1.q_values(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "dqn_head_ckpt")
        head1.save(path)

        head2 = DQNHead(
            obs_dim=4,
            n_actions=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            dueling_mode=True,
            device="cpu",
        )
        head2.load(path)

        sd1 = head1.q.state_dict()
        sd2 = head2.q.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"online Q mismatch after load: {k}")

        td1 = head1.q_target.state_dict()
        td2 = head2.q_target.state_dict()
        assert_eq(set(td1.keys()), set(td2.keys()))
        for k in td1.keys():
            assert_true(th.allclose(td1[k], td2[k]), f"target Q mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration
# =============================================================================
def test_dqn_on_env_step_requires_standard_fields():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=20, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)
    a = int(env.action_space.sample())
    next_obs, r, terminated, truncated, _ = env.step(a)
    done = bool(terminated or truncated)

    transition_missing = {
        "obs": obs,
        "action": a,
        "reward": float(r),
        "done": done,
        # "next_obs" missing
    }

    # Your on_env_step() currently accesses transition["next_obs"] directly,
    # so it may raise KeyError. If you later add explicit validation,
    # this will become ValueError.
    assert_raises((KeyError, ValueError), lambda: algo.on_env_step(transition_missing))
    env.close()


def test_dqn_smoke_run_no_per_runs_and_updates():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    out = _rollout_steps(algo, env, n_steps=1500, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")

    assert_true("loss/q" in m, "missing loss/q")
    assert_true("q/mean" in m, "missing q/mean")
    assert_true("target/mean" in m, "missing target/mean")
    assert_true(np.isfinite(float(m["loss/q"])), "loss/q not finite")

    # Optional but recommended if you added epsilon logging:
    # assert_true("exploration/epsilon" in m, "missing exploration/epsilon")


def test_dqn_smoke_run_with_per_reports_beta():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=True)
    out = _rollout_steps(algo, env, n_steps=1800, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")

    # DQNCore returns td_errors vector for PER feedback.
    # If your OffPolicyAlgorithm pops it internally, it won't appear in m.
    # So this check is intentionally soft.
    # (If you want it surfaced, do not pop it before filtering, or log separately.)


def test_dqn_replay_batch_contract():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    _ = _rollout_steps(algo, env, n_steps=400, seed=0, do_updates=False)

    assert_true(algo.buffer is not None, "buffer should be initialized")
    assert_true(algo.buffer.size >= algo.batch_size, "buffer must have enough samples")

    batch = algo.buffer.sample(algo.batch_size)

    assert_true(hasattr(batch, "observations"), "batch missing observations")
    assert_true(hasattr(batch, "actions"), "batch missing actions")
    assert_true(hasattr(batch, "rewards"), "batch missing rewards")
    assert_true(hasattr(batch, "next_observations"), "batch missing next_observations")
    assert_true(hasattr(batch, "dones"), "batch missing dones")

    if th.is_tensor(batch.observations):
        assert_eq(tuple(batch.observations.shape), (algo.batch_size, 4))
    if th.is_tensor(batch.next_observations):
        assert_eq(tuple(batch.next_observations.shape), (algo.batch_size, 4))

    if th.is_tensor(batch.actions):
        shp = tuple(batch.actions.shape)
        assert_true(shp == (algo.batch_size,) or shp == (algo.batch_size, 1), f"actions shape bad: {shp}")

    env.close()


def test_dqn_parameters_change_after_update():
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_int = int(np.asarray(a).reshape(-1)[0]) if isinstance(a, (np.ndarray, list, tuple)) else int(a)

        next_obs, r, terminated, truncated, _ = env.step(a_int)
        done = bool(terminated or truncated)

        algo.on_env_step({"obs": obs, "action": a_int, "reward": float(r), "next_obs": next_obs, "done": done})

        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    q_before = _state_vector(algo.head.q)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    q_after = _state_vector(algo.head.q)
    dq = float(th.linalg.vector_norm(q_after - q_before).item()) if q_before.numel() else 0.0

    assert_true(dq > 0.0, f"expected Q params to change; dq={dq}")
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("dqn_head_q_values_and_act_shapes", test_dqn_head_q_values_and_act_shapes),
    ("dqn_head_save_and_load_roundtrip", test_dqn_head_save_and_load_roundtrip),
    ("dqn_on_env_step_requires_standard_fields", test_dqn_on_env_step_requires_standard_fields),
    ("dqn_smoke_run_no_per_runs_and_updates", test_dqn_smoke_run_no_per_runs_and_updates),
    ("dqn_smoke_run_with_per_reports_beta", test_dqn_smoke_run_with_per_reports_beta),
    ("dqn_replay_batch_contract", test_dqn_replay_batch_contract),
    ("dqn_parameters_change_after_update", test_dqn_parameters_change_after_update),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="dqn")


if __name__ == "__main__":
    raise SystemExit(main())
