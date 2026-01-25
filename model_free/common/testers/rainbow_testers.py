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
    Make imports work when running this file directly (without installing as a package).

    Strategy:
      - Walk up parent directories looking for a top-level "model_free" folder.
      - If found, prepend that parent to sys.path.
      - Otherwise, fall back to <repo_root_guess>.
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

# NOTE: Update these import paths if your repo layout differs.
from model_free.baselines.q_learning.rainbow.rainbow import rainbow  # noqa: E402
from model_free.baselines.q_learning.rainbow.head import RainbowHead  # noqa: E402


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
            self.__post_init__()  # re-init rng/spaces deterministically
        self._t = 0
        self._obs = self.observation_space.sample() * 0.0  # start near zero
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)

        # Simple deterministic-ish dynamics with small noise
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
    """
    Flatten all parameters into a single CPU float vector (for change detection).
    """
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _make_small_algo(*, device: str = "cpu", use_per: bool = True, n_step: int = 1) -> Any:
    """
    Create a small Rainbow algo for fast unit tests.

    Important:
      - warmup_env_steps/update_after are kept small so tests finish quickly.
      - batch_size is small to reduce compute.
    """
    return rainbow(
        obs_dim=4,
        n_actions=2,
        device=device,
        # head
        atom_size=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_sizes=(64, 64),
        activation_fn=th.nn.ReLU,
        noisy_std_init=0.5,
        # core
        gamma=0.99,
        n_step=int(n_step),
        double_dqn=True,
        target_update_interval=50,
        tau=0.0,
        max_grad_norm=10.0,
        use_amp=False,
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
        per_eps=1e-6,
        # exploration schedule (Rainbow typically uses eps ~ 0, but we keep it non-zero to exercise plumbing)
        exploration_eps=0.1,
        exploration_eps_final=0.05,
        exploration_eps_anneal_steps=1_000,
        exploration_eval_eps=0.0,
    )


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Rollout transitions into OffPolicyAlgorithm and optionally run updates.

    Assumptions:
      - algo.act(obs, deterministic=...) does NOT accept epsilon directly.
        (epsilon should be handled internally by algo via its exploration schedule)
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        # For Rainbow, deterministic=False is typical during training.
        a = algo.act(obs, deterministic=False)
        a_int = int(a) if not th.is_tensor(a) else int(a.detach().cpu().item())

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
# Tests: RainbowHead
# =============================================================================
def test_rainbow_head_dist_and_act_shapes() -> None:
    seed_all(0)

    head = RainbowHead(
        obs_dim=4,
        n_actions=2,
        atom_size=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        noisy_std_init=0.5,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # dist: (B,A,K) = (1,2,51)
    with th.no_grad():
        s = head._to_tensor_batched(obs)  # (1,4)
        dist = head.q.dist(s)
    assert_true(th.is_tensor(dist), "head.q.dist should return torch.Tensor")
    assert_eq(tuple(dist.shape), (1, 2, 51))

    # expected Q: (B,A)
    q = head.q_values(obs)
    assert_true(th.is_tensor(q), "head.q_values should return torch.Tensor")
    assert_eq(tuple(q.shape), (1, 2))

    # act: (B,) for single obs -> (1,)
    a = head.act(obs, epsilon=0.0, deterministic=True)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1,))


def test_rainbow_head_save_and_load_roundtrip() -> None:
    seed_all(0)

    head1 = RainbowHead(
        obs_dim=4,
        n_actions=2,
        atom_size=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        noisy_std_init=0.5,
        device="cpu",
    )

    _ = head1.q_values(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rainbow_head_ckpt")
        head1.save(path)

        head2 = RainbowHead(
            obs_dim=4,
            n_actions=2,
            atom_size=51,
            v_min=-10.0,
            v_max=10.0,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            noisy_std_init=0.5,
            device="cpu",
        )
        head2.load(path)

        sd1 = head1.q.state_dict()
        sd2 = head2.q.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"online q mismatch after load: {k}")


# =============================================================================
# Tests: OffPolicyAlgorithm integration
# =============================================================================
def test_rainbow_on_env_step_requires_standard_fields() -> None:
    """
    OffPolicyAlgorithm.on_env_step currently indexes required keys directly.
    Therefore, missing fields commonly raise KeyError (unless the implementation
    explicitly validates and raises ValueError).

    This test locks in the current contract for required keys:
      - obs, action, reward, next_obs, done
    """
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
        # "next_obs": next_obs,  # missing
        "done": done,
    }

    # Expect KeyError given direct dict indexing in on_env_step().
    assert_raises(KeyError, lambda: algo.on_env_step(transition_missing))
    env.close()


def test_rainbow_smoke_run_no_per_runs_and_updates() -> None:
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    out = _rollout_steps(algo, env, n_steps=1500, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # Standard offpolicy bookkeeping
    assert_true("offpolicy/buffer_size" in m, "missing offpolicy/buffer_size")
    assert_true("offpolicy/env_steps" in m, "missing offpolicy/env_steps")

    # Core scalar metrics (filtered)
    assert_true("loss/q" in m, "missing loss/q")
    assert_true("q/mean" in m, "missing q/mean")
    assert_true(np.isfinite(float(m["loss/q"])), "loss/q not finite")
    assert_true(np.isfinite(float(m["q/mean"])), "q/mean not finite")


def test_rainbow_smoke_run_with_per_reports_beta() -> None:
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=True)
    out = _rollout_steps(algo, env, n_steps=1800, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run (PER)")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # PER bookkeeping emitted by OffPolicyAlgorithm.update_once()
    assert_true("per/enabled" in m, "missing per/enabled")
    assert_true("per/beta" in m, "missing per/beta")
    assert_true(0.0 <= float(m["per/beta"]) <= 1.0, "per/beta out of range [0,1]")


def test_rainbow_replay_batch_contract() -> None:
    """
    Validate replay batch fields needed by RainbowCore.

    RainbowCore consumes:
      - observations, actions, rewards, dones, next_observations
    and optionally:
      - n_step_returns, n_step_dones, n_step_next_observations (if n_step>1 and replay provides them)
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    # Exercise both n_step paths; if replay doesn't implement n-step fields,
    # core falls back gracefully, so we only assert the mandatory fields.
    algo = _make_small_algo(device="cpu", use_per=False, n_step=3)
    algo.setup(env)

    _ = _rollout_steps(algo, env, n_steps=400, seed=0, do_updates=False)

    assert_true(algo.buffer is not None, "buffer should be initialized")
    assert_true(algo.buffer.size >= algo.batch_size, "buffer must have enough samples")

    batch = algo.buffer.sample(algo.batch_size)

    # Mandatory fields
    assert_true(hasattr(batch, "observations"), "batch missing observations")
    assert_true(hasattr(batch, "actions"), "batch missing actions")
    assert_true(hasattr(batch, "rewards"), "batch missing rewards")
    assert_true(hasattr(batch, "dones"), "batch missing dones")
    assert_true(hasattr(batch, "next_observations"), "batch missing next_observations")

    # Optional n-step fields (best-effort; do not fail if absent)
    if hasattr(batch, "n_step_returns") and hasattr(batch, "n_step_dones") and hasattr(batch, "n_step_next_observations"):
        r = batch.n_step_returns
        d = batch.n_step_dones
        ns = batch.n_step_next_observations
        if th.is_tensor(r):
            assert_true(int(r.shape[0]) == int(algo.batch_size), f"n_step_returns bad shape: {tuple(r.shape)}")
        if th.is_tensor(d):
            assert_true(int(d.shape[0]) == int(algo.batch_size), f"n_step_dones bad shape: {tuple(d.shape)}")
        if th.is_tensor(ns):
            assert_true(int(ns.shape[0]) == int(algo.batch_size), f"n_step_next_observations bad shape: {tuple(ns.shape)}")

    env.close()


def test_rainbow_parameters_change_after_update() -> None:
    """
    Ensure that at least one parameter in the online network changes after update().
    """
    seed_all(0)
    env = DummyCartPoleLikeEnv(horizon=50, seed=0)

    algo = _make_small_algo(device="cpu", use_per=False)
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_int = int(a) if not th.is_tensor(a) else int(a.detach().cpu().item())

        next_obs, r, terminated, truncated, _ = env.step(a_int)
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

        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    # Compare online network parameters (head.q)
    q_before = _state_vector(algo.head.q)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    q_after = _state_vector(algo.head.q)

    dq = float(th.linalg.vector_norm(q_after - q_before).item()) if q_before.numel() else 0.0
    assert_true(dq > 0.0, f"expected q params to change; dq={dq}")

    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("rainbow_head_dist_and_act_shapes", test_rainbow_head_dist_and_act_shapes),
    ("rainbow_head_save_and_load_roundtrip", test_rainbow_head_save_and_load_roundtrip),
    ("rainbow_on_env_step_requires_standard_fields", test_rainbow_on_env_step_requires_standard_fields),
    ("rainbow_smoke_run_no_per_runs_and_updates", test_rainbow_smoke_run_no_per_runs_and_updates),
    ("rainbow_smoke_run_with_per_reports_beta", test_rainbow_smoke_run_with_per_reports_beta),
    ("rainbow_replay_batch_contract", test_rainbow_replay_batch_contract),
    ("rainbow_parameters_change_after_update", test_rainbow_parameters_change_after_update),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="rainbow")


if __name__ == "__main__":
    raise SystemExit(main())
