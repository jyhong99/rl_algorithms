# test_vpg.py
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


# =============================================================================
# sys.path bootstrap (run-as-script friendly)
# =============================================================================
def _bootstrap_sys_path() -> None:
    """
    Add project root to sys.path so tests can be run directly as a script.

    Strategy
    --------
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
    assert_raises,
)

from model_free.baselines.policy_gradients.on_policy.vpg.vpg import vpg  # noqa: E402
from model_free.baselines.policy_gradients.on_policy.vpg.head import VPGHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
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
class DummyContinuousEpisodicEnv:
    """
    Tiny episodic continuous-control env for smoke/integration tests.

    Shapes
    ------
    - obs: (obs_dim,) float32
    - act: (act_dim,) float32

    Dynamics (toy)
    --------------
    - next_obs = clip(obs + 0.05 * action + noise, -1, 1)
    - reward   = 1.0 per step (dense), for wiring tests only
    - episode ends by horizon only (truncated=True at horizon)

    Not intended for meaningful learning; only to exercise algorithm plumbing.
    """

    obs_dim: int = 4
    act_dim: int = 2
    horizon: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        self.observation_space = DummyBoxSpace(shape=(self.obs_dim,), low=-1.0, high=1.0, seed=self.seed)
        self.action_space = DummyBoxSpace(shape=(self.act_dim,), low=-1.0, high=1.0, seed=self.seed + 123)
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        noise = self._rng.randn(self.obs_dim).astype(np.float32) * 0.01

        # Map action_dim -> obs_dim (simple projection/repeat)
        if self.act_dim == self.obs_dim:
            delta = 0.05 * a
        else:
            reps = int(np.ceil(self.obs_dim / self.act_dim))
            delta = 0.05 * np.tile(a, reps)[: self.obs_dim]

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
    """Flatten all parameters into a single CPU float vector for change detection."""
    parts: List[th.Tensor] = []
    for p in module.parameters(recurse=True):
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Roll out n_steps transitions and optionally run updates when ready.

    Contract
    --------
    - algo.setup(env)
    - loop:
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
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.act_dim)

        next_obs, r, terminated, truncated, _info = env.step(a_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a_np,
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
# Tests: VPGHead
# =============================================================================
def test_vpg_head_act_and_evaluate_shapes_baseline_on():
    """
    Baseline ON: head should provide actor + critic, and evaluate_actions should include value.
    """
    seed_all(0)

    head = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=True,
    )

    obs = np.zeros((4,), dtype=np.float32)
    a = head.act(obs, deterministic=False)

    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1, 2))

    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    assert_true("log_prob" in out and "entropy" in out, "missing log_prob/entropy in evaluate_actions output")

    # Value expected when baseline is enabled
    assert_true("value" in out, "baseline enabled: expected 'value' in evaluate_actions output")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]

    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "value/log_prob/entropy must be tensors")
    assert_eq(tuple(v.shape), (1, 1))
    assert_true(lp.shape[0] == 1, f"log_prob batch mismatch: {tuple(lp.shape)}")
    assert_true(ent.shape[0] == 1, f"entropy batch mismatch: {tuple(ent.shape)}")


def test_vpg_head_act_and_evaluate_shapes_baseline_off_tolerant():
    """
    Baseline OFF: head has actor only. evaluate_actions behavior depends on base head design.

    This test is tolerant:
    - It requires log_prob and entropy.
    - If 'value' exists, it must be tensor-shaped (B,1) or batch-aligned.
      Otherwise, absence of 'value' is also acceptable.
    """
    seed_all(0)

    head = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=False,
    )

    obs = np.zeros((4,), dtype=np.float32)
    a = head.act(obs, deterministic=False)

    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1, 2))

    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    assert_true("log_prob" in out and "entropy" in out, "missing log_prob/entropy in evaluate_actions output")

    lp = out["log_prob"]
    ent = out["entropy"]
    assert_true(th.is_tensor(lp) and th.is_tensor(ent), "log_prob/entropy must be tensors")
    assert_true(lp.shape[0] == 1, f"log_prob batch mismatch: {tuple(lp.shape)}")
    assert_true(ent.shape[0] == 1, f"entropy batch mismatch: {tuple(ent.shape)}")

    if "value" in out:
        v = out["value"]
        assert_true(th.is_tensor(v), "if present, value must be a tensor")
        assert_true(v.shape[0] == 1, f"value batch mismatch: {tuple(v.shape)}")


def test_vpg_head_save_and_load_roundtrip_baseline_on():
    """Baseline ON: save/load should restore identical actor/critic weights."""
    seed_all(0)

    head1 = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=True,
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vpg_head_on_ckpt")
        head1.save(path)

        head2 = VPGHead(
            obs_dim=4,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
            log_std_mode="param",
            log_std_init=-0.5,
            use_baseline=True,
        )
        head2.load(path)

        # Actor
        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        # Critic
        assert_true(head1.critic is not None and head2.critic is not None, "critics should exist (baseline on)")
        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


def test_vpg_head_save_and_load_roundtrip_baseline_off():
    """Baseline OFF: save/load should restore identical actor weights and critic should remain None."""
    seed_all(0)

    head1 = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=False,
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vpg_head_off_ckpt")
        head1.save(path)

        head2 = VPGHead(
            obs_dim=4,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            device="cpu",
            log_std_mode="param",
            log_std_init=-0.5,
            use_baseline=False,
        )
        head2.load(path)

        sd1a = head1.actor.state_dict()
        sd2a = head2.actor.state_dict()
        assert_eq(set(sd1a.keys()), set(sd2a.keys()))
        for k in sd1a.keys():
            assert_true(th.allclose(sd1a[k], sd2a[k]), f"actor mismatch after load: {k}")

        assert_true(head2.critic is None, "baseline off: critic should be None")


def test_vpg_head_baseline_mismatch_raises():
    """Loading baseline-on checkpoint into baseline-off head (or vice versa) should raise."""
    seed_all(0)

    head_on = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=True,
    )
    head_off = VPGHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
        use_baseline=False,
    )

    with tempfile.TemporaryDirectory() as td:
        p_on = os.path.join(td, "vpg_on")
        p_off = os.path.join(td, "vpg_off")
        head_on.save(p_on)
        head_off.save(p_off)

        # ON -> OFF should fail (checkpoint has critic weights)
        assert_raises(ValueError, lambda: head_off.load(p_on))

        # OFF -> ON should fail (checkpoint has no critic weights)
        assert_raises(ValueError, lambda: head_on.load(p_off))


# =============================================================================
# Tests: OnPolicyAlgorithm integration (VPG)
# =============================================================================
def test_vpg_algorithm_smoke_run_runs_and_updates_baseline_on():
    """End-to-end smoke test (baseline on): rollout until update triggers; validate metrics."""
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = vpg(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        use_baseline=True,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,  # full batch
        actor_lr=3e-4,
        critic_lr=3e-4,
        ent_coef=0.0,
        vf_coef=0.5,
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one VPG update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    for k in [
        "loss/policy",
        "loss/entropy",
        "loss/total",
        "stats/entropy",
        "lr/actor",
        "loss/value",
        "stats/value_mean",
        "lr/critic",
    ]:
        assert_true(k in m, f"missing metric: {k}")

    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")
    assert_true(np.isfinite(float(m["loss/value"])), "loss/value not finite")


def test_vpg_algorithm_smoke_run_runs_and_updates_baseline_off():
    """End-to-end smoke test (baseline off): actor-only updates; validate metrics."""
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = vpg(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        use_baseline=False,
        hidden_sizes=(64, 64),
        rollout_steps=256,
        update_epochs=1,
        minibatch_size=None,
        actor_lr=3e-4,
        ent_coef=0.0,
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one VPG update to run")
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

    # Baseline off: these should NOT exist (per your VPGCore)
    assert_true("loss/value" not in m, "baseline off: did not expect loss/value")
    assert_true("lr/critic" not in m, "baseline off: did not expect lr/critic")

    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")


def test_vpg_parameters_change_after_update_baseline_on():
    """Baseline ON: ensure at least one of actor/critic parameters changes after update()."""
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = vpg(
        obs_dim=4,
        action_dim=2,
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
        a_np = np.asarray(a, dtype=np.float32).reshape(env.act_dim)

        next_obs, r, terminated, truncated, _ = env.step(a_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {"obs": obs, "action": a_np, "reward": float(r), "next_obs": next_obs, "done": done}
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
        drove += 1

    assert_true(algo.ready_to_update(), "algo did not become ready_to_update() within drive budget")

    actor_before = _state_vector(algo.head.actor)
    critic_before = _state_vector(algo.head.critic) if getattr(algo.head, "critic", None) is not None else th.zeros(0)

    m = algo.update()
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return metrics")

    actor_after = _state_vector(algo.head.actor)
    critic_after = _state_vector(algo.head.critic) if getattr(algo.head, "critic", None) is not None else th.zeros(0)

    da = float(th.linalg.vector_norm(actor_after - actor_before).item()) if actor_before.numel() else 0.0
    dc = float(th.linalg.vector_norm(critic_after - critic_before).item()) if critic_before.numel() else 0.0

    assert_true((da > 0.0) or (dc > 0.0), f"expected params to change; da={da}, dc={dc}")
    env.close()


def test_vpg_parameters_change_after_update_baseline_off():
    """Baseline OFF: ensure actor parameters change after update()."""
    seed_all(0)
    env = DummyContinuousEpisodicEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = vpg(
        obs_dim=4,
        action_dim=2,
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

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        a_np = np.asarray(a, dtype=np.float32).reshape(env.act_dim)

        next_obs, r, terminated, truncated, _ = env.step(a_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {"obs": obs, "action": a_np, "reward": float(r), "next_obs": next_obs, "done": done}
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
    ("vpg_head_act_and_evaluate_shapes_baseline_on", test_vpg_head_act_and_evaluate_shapes_baseline_on),
    ("vpg_head_act_and_evaluate_shapes_baseline_off_tolerant", test_vpg_head_act_and_evaluate_shapes_baseline_off_tolerant),
    ("vpg_head_save_and_load_roundtrip_baseline_on", test_vpg_head_save_and_load_roundtrip_baseline_on),
    ("vpg_head_save_and_load_roundtrip_baseline_off", test_vpg_head_save_and_load_roundtrip_baseline_off),
    ("vpg_head_baseline_mismatch_raises", test_vpg_head_baseline_mismatch_raises),
    ("vpg_algorithm_smoke_run_runs_and_updates_baseline_on", test_vpg_algorithm_smoke_run_runs_and_updates_baseline_on),
    ("vpg_algorithm_smoke_run_runs_and_updates_baseline_off", test_vpg_algorithm_smoke_run_runs_and_updates_baseline_off),
    ("vpg_parameters_change_after_update_baseline_on", test_vpg_parameters_change_after_update_baseline_on),
    ("vpg_parameters_change_after_update_baseline_off", test_vpg_parameters_change_after_update_baseline_off),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="vpg")


if __name__ == "__main__":
    raise SystemExit(main())
