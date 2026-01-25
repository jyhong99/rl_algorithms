# vpg_discrete_testers.py
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
    Add project root to sys.path so tests can be run directly as a script.

    Mirrors your existing tester pattern:
      - Walk parents up to a fixed depth to find a folder containing "model_free".
      - Insert that parent directory into sys.path.
      - Fallback to a reasonable default if not found.
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
)

from model_free.baselines.policy_gradients.on_policy.vpg_discrete.vpg_discrete import (  # noqa: E402
    vpg_discrete,
)
from model_free.baselines.policy_gradients.on_policy.vpg_discrete.head import (  # noqa: E402
    VPGDiscreteHead,
)


# =============================================================================
# Minimal discrete spaces/env (no gym dependency)
# =============================================================================
class DummyDiscreteSpace:
    """Minimal discrete action space with sample()."""

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
class DummyDiscreteEpisodicEnv:
    """
    A tiny episodic discrete-control env (gym-free).

    Shapes
    ------
    - obs: (obs_dim,) float32
    - act: int in [0, n_actions)

    Dynamics (toy)
    --------------
    - next_obs = clip(obs + delta(action) + noise, -1, 1)
    - reward   = +1.0 per step
    - episode terminates by horizon only (truncated=True at horizon)

    Purpose
    -------
    This environment is intentionally simple and deterministic-ish to exercise:
    - algo.setup / act / on_env_step wiring
    - ready_to_update scheduling
    - update() returning metrics with expected keys
    """

    obs_dim: int = 4
    n_actions: int = 3
    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(self.obs_dim,), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyDiscreteSpace(n=self.n_actions, seed=self.seed + 123)
        self._rng = np.random.RandomState(self.seed + 999)
        self._t = 0
        self._obs = np.zeros((self.obs_dim,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed = int(seed)
            self.__post_init__()
        self._t = 0
        self._obs = np.zeros((self.obs_dim,), dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = int(action)
        noise = self._rng.randn(self.obs_dim).astype(np.float32) * 0.01

        # Simple deterministic-ish action effect: shift one coordinate
        delta = np.zeros((self.obs_dim,), dtype=np.float32)
        delta[a % self.obs_dim] = 0.05

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
    Flatten all parameters into a single CPU float vector for "did params change?" checks.
    """
    parts: List[th.Tensor] = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Roll out n_steps transitions and optionally run updates when ready.

    Contract (matches your other testers)
    -------------------------------------
      - algo.setup(env)
      - for each step:
          a = algo.act(obs)
          env.step(a)
          algo.on_env_step(transition)
          if algo.ready_to_update(): algo.update()
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        # Discrete action: return may be tensor or int depending on head/algo
        a = algo.act(obs, deterministic=False)

        if th.is_tensor(a):
            a_item = int(a.detach().cpu().reshape(-1)[0].item())
        else:
            a_item = int(np.asarray(a).reshape(-1)[0])

        next_obs, r, terminated, truncated, _info = env.step(a_item)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_item,  # store as int (algo dtype_act=np.int64)
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
# Tests: VPGDiscreteHead
# =============================================================================
def test_vpg_discrete_head_act_and_evaluate_shapes_baseline_on():
    """
    Basic head interface test (baseline ON):
      - head.act returns a tensor action (B,) or (B,1) depending on base head
      - head.evaluate_actions returns value/log_prob/entropy with (B,1) shapes
    """
    seed_all(0)

    head = VPGDiscreteHead(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        use_baseline=True,
    )

    obs = np.zeros((4,), dtype=np.float32)

    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")

    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys in evaluate_actions output")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]

    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "value/log_prob/entropy must be tensors")
    assert_eq(tuple(v.shape), (1, 1))
    assert_eq(tuple(lp.shape), (1, 1))
    assert_eq(tuple(ent.shape), (1, 1))


def test_vpg_discrete_head_act_and_evaluate_shapes_baseline_off():
    """
    Basic head interface test (baseline OFF):
      - head.evaluate_actions must NOT crash even when critic is None
      - value should be returned as zero baseline with shape (B,1)
    """
    seed_all(0)

    head = VPGDiscreteHead(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        use_baseline=False,
    )

    obs = np.zeros((4,), dtype=np.float32)

    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")

    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    v = out["value"]
    assert_true(th.is_tensor(v), "value must be a tensor")
    assert_eq(tuple(v.shape), (1, 1))
    assert_true(float(v.detach().cpu().item()) == 0.0, "baseline-off value should be zero")


def test_vpg_discrete_head_save_and_load_roundtrip_baseline_on():
    """
    Save/load should restore identical actor/critic weights (baseline ON).
    """
    seed_all(0)

    head1 = VPGDiscreteHead(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        use_baseline=True,
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vpg_discrete_head_ckpt")
        head1.save(path)

        head2 = VPGDiscreteHead(
            obs_dim=4,
            n_actions=3,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
            use_baseline=True,
        )
        head2.load(path)

        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        assert_true(head1.critic is not None and head2.critic is not None, "critics must exist for baseline ON")
        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


def test_vpg_discrete_head_save_and_load_roundtrip_baseline_off():
    """
    Save/load should restore identical actor weights and preserve critic=None (baseline OFF).
    """
    seed_all(0)

    head1 = VPGDiscreteHead(
        obs_dim=4,
        n_actions=3,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        use_baseline=False,
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vpg_discrete_head_ckpt")
        head1.save(path)

        head2 = VPGDiscreteHead(
            obs_dim=4,
            n_actions=3,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
            use_baseline=False,
        )
        head2.load(path)

        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        assert_true(head2.critic is None, "baseline-off head must keep critic=None after load")


# =============================================================================
# Tests: OnPolicyAlgorithm integration (VPGDiscrete)
# =============================================================================
def test_vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_on():
    """
    End-to-end smoke test (baseline ON):
      - rollout until update triggers
      - ensure update() returns metrics with expected keys
    """
    seed_all(0)
    env = DummyDiscreteEpisodicEnv(obs_dim=4, n_actions=3, horizon=50, seed=0)

    algo = vpg_discrete(
        obs_dim=4,
        n_actions=3,
        device="cpu",
        use_baseline=True,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,
        actor_lr=3e-4,
        critic_lr=3e-4,
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one VPGDiscrete update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    for k in [
        "loss/policy",
        "loss/value",
        "loss/entropy",
        "loss/total",
        "stats/entropy",
        "stats/value_mean",
        "lr/actor",
        "lr/critic",
    ]:
        assert_true(k in m, f"missing metric: {k}")

    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")


def test_vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_off():
    """
    End-to-end smoke test (baseline OFF):
      - rollout until update triggers
      - ensure update() returns metrics (without value/critic keys)
    """
    seed_all(0)
    env = DummyDiscreteEpisodicEnv(obs_dim=4, n_actions=3, horizon=50, seed=0)

    algo = vpg_discrete(
        obs_dim=4,
        n_actions=3,
        device="cpu",
        use_baseline=False,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,
        actor_lr=3e-4,
        critic_lr=3e-4,  # ignored
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one VPGDiscrete update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    for k in [
        "loss/policy",
        "loss/entropy",
        "loss/total",
        "stats/entropy",
        "lr/actor",
    ]:
        assert_true(k in m, f"missing metric: {k}")

    # Baseline-off must not require critic metrics
    assert_true("loss/value" not in m, "baseline-off should not report value loss")
    assert_true("lr/critic" not in m, "baseline-off should not report critic lr")

    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")


def test_vpg_discrete_parameters_change_after_update_baseline_on():
    """
    Verify that actor (and critic) parameters change after at least one update (baseline ON).
    """
    seed_all(0)
    env = DummyDiscreteEpisodicEnv(obs_dim=4, n_actions=3, horizon=50, seed=0)

    algo = vpg_discrete(
        obs_dim=4,
        n_actions=3,
        device="cpu",
        use_baseline=True,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,
        actor_lr=3e-4,
        critic_lr=3e-4,
        use_amp=False,
    )
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    # Drive until ready_to_update() becomes true
    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        if th.is_tensor(a):
            a_item = int(a.detach().cpu().reshape(-1)[0].item())
        else:
            a_item = int(np.asarray(a).reshape(-1)[0])

        next_obs, r, terminated, truncated, _ = env.step(a_item)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_item,
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
    critic_before = _state_vector(algo.head.critic)  # type: ignore[union-attr]

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    actor_after = _state_vector(algo.head.actor)
    critic_after = _state_vector(algo.head.critic)  # type: ignore[union-attr]

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    dc = float(th.linalg.vector_norm(critic_after - critic_before).item()) if critic_before.numel() else 0.0

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")
    env.close()


def test_vpg_discrete_parameters_change_after_update_baseline_off():
    """
    Verify that actor parameters change after at least one update (baseline OFF).
    """
    seed_all(0)
    env = DummyDiscreteEpisodicEnv(obs_dim=4, n_actions=3, horizon=50, seed=0)

    algo = vpg_discrete(
        obs_dim=4,
        n_actions=3,
        device="cpu",
        use_baseline=False,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,
        actor_lr=3e-4,
        use_amp=False,
    )
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    # Drive until ready_to_update() becomes true
    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        if th.is_tensor(a):
            a_item = int(a.detach().cpu().reshape(-1)[0].item())
        else:
            a_item = int(np.asarray(a).reshape(-1)[0])

        next_obs, r, terminated, truncated, _ = env.step(a_item)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_item,
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

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    actor_after = _state_vector(algo.head.actor)

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    assert_true(da > 0.0, f"expected actor params to change; da={da}")
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("vpg_discrete_head_act_and_evaluate_shapes_baseline_on", test_vpg_discrete_head_act_and_evaluate_shapes_baseline_on),
    ("vpg_discrete_head_act_and_evaluate_shapes_baseline_off", test_vpg_discrete_head_act_and_evaluate_shapes_baseline_off),
    ("vpg_discrete_head_save_and_load_roundtrip_baseline_on", test_vpg_discrete_head_save_and_load_roundtrip_baseline_on),
    ("vpg_discrete_head_save_and_load_roundtrip_baseline_off", test_vpg_discrete_head_save_and_load_roundtrip_baseline_off),
    ("vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_on", test_vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_on),
    ("vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_off", test_vpg_discrete_algorithm_smoke_run_runs_and_updates_baseline_off),
    ("vpg_discrete_parameters_change_after_update_baseline_on", test_vpg_discrete_parameters_change_after_update_baseline_on),
    ("vpg_discrete_parameters_change_after_update_baseline_off", test_vpg_discrete_parameters_change_after_update_baseline_off),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="vpg_discrete")


if __name__ == "__main__":
    raise SystemExit(main())
