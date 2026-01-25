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
    Make local repo importable when running this file directly.

    Walk up a few parents looking for a `model_free/` directory, and prepend that
    parent to sys.path. Falls back to two-level parent of this file.
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

from model_free.baselines.policy_gradients.on_policy.acktr.acktr import acktr  # noqa: E402
from model_free.baselines.policy_gradients.on_policy.acktr.head import ACKTRHead  # noqa: E402


# =============================================================================
# Minimal spaces/env (no gym dependency)
# =============================================================================
class DummyBoxSpace:
    """Minimal Box(space) with sample()."""

    def __init__(self, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.low = np.full(self.shape, float(low), dtype=np.float32)
        self.high = np.full(self.shape, float(high), dtype=np.float32)
        self._rng = np.random.RandomState(int(seed))

    def sample(self) -> np.ndarray:
        return self._rng.uniform(low=self.low, high=self.high).astype(np.float32)


@dataclass
class DummyContinuousEnv:
    """
    A tiny episodic continuous env.

    Shapes:
      - obs: (obs_dim,) float32
      - action: (act_dim,) float32

    Episode ends after horizon steps.
    Reward = 1.0 per step.
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
        self._obs = self.observation_space.sample() * 0.0
        return self._obs.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        noise = self._rng.randn(self.obs_dim).astype(np.float32) * 0.01
        # simple dynamics: obs += affine(action) + noise
        delta = 0.05 * float(np.tanh(a.mean()))
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
    """Flatten all parameters into a single CPU float vector."""
    parts = []
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        parts.append(p.detach().reshape(-1).cpu().float())
    if not parts:
        return th.zeros((0,), dtype=th.float32)
    return th.cat(parts, dim=0)


def _as_float_action(act: Any, act_dim: int) -> np.ndarray:
    """
    Normalize action into np.float32 shape (act_dim,).

    This is defensive because different policies may return:
      - np.ndarray (act_dim,)
      - torch.Tensor (act_dim,) or (1, act_dim)
      - python float for act_dim==1
    """
    if th.is_tensor(act):
        a = act.detach().cpu().numpy()
    else:
        a = np.asarray(act)

    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        # scalar -> (1,) then broadcast/reshape to (act_dim,)
        a = a.reshape(1)
    if a.ndim == 1:
        if a.shape[0] == act_dim:
            return a
        if a.shape[0] == 1 and act_dim == 1:
            return a.reshape(act_dim)
    if a.ndim == 2:
        # common: (B, act_dim) with B=1
        if a.shape[0] == 1 and a.shape[1] == act_dim:
            return a.reshape(act_dim)

    raise ValueError(f"Unexpected action shape for continuous env: {a.shape} (act_dim={act_dim})")


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Run env interaction loop and optionally call algo.update() when ready.

    This assumes OnPolicyAlgorithm collects transitions via on_env_step(...)
    and resolves value/logp using head.evaluate_actions internally.
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        act = algo.act(obs, deterministic=False)
        act_np = _as_float_action(act, env.act_dim)

        next_obs, r, terminated, truncated, _info = env.step(act_np)
        done = bool(terminated or truncated)

        # Minimal transition contract for OnPolicyAlgorithm (typical)
        algo.on_env_step(
            {
                "obs": obs,
                "action": act_np,
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
# Tests: ACKTRHead
# =============================================================================
def test_acktr_head_actor_dist_and_evaluate_shapes():
    seed_all(0)

    head = ACKTRHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
    )

    obs = np.zeros((4,), dtype=np.float32)

    # act() should return an action tensor; shape normalization is handled by base head
    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    # allow (act_dim,) or (1, act_dim) depending on your base head implementation
    assert_true(tuple(a.shape) in [(2,), (1, 2)], f"unexpected action shape: {tuple(a.shape)}")

    # evaluate_actions should return value/log_prob/entropy
    out = head.evaluate_actions(obs, a)
    assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys in evaluate_actions output")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]
    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "value/log_prob/entropy must be tensors")

    # We expect standardized shapes (B,1) from base head; but accept (1,) as fallback
    assert_true(tuple(v.shape) in [(1, 1), (1,)], f"value shape unexpected: {tuple(v.shape)}")
    assert_true(tuple(lp.shape) in [(1, 1), (1,)], f"log_prob shape unexpected: {tuple(lp.shape)}")
    assert_true(tuple(ent.shape) in [(1, 1), (1,)], f"entropy shape unexpected: {tuple(ent.shape)}")


def test_acktr_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = ACKTRHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        device="cpu",
        log_std_mode="param",
        log_std_init=-0.5,
    )

    # touch modules to ensure params exist and any lazy buffers are initialized
    _ = head1.act(np.zeros((4,), dtype=np.float32), deterministic=False)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "acktr_head_ckpt")
        head1.save(path)

        head2 = ACKTRHead(
            obs_dim=4,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            init_type="orthogonal",
            gain=1.0,
            bias=0.0,
            device="cpu",
            log_std_mode="param",
            log_std_init=-0.5,
        )
        head2.load(path)

        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")

        cd1 = head1.critic.state_dict()
        cd2 = head2.critic.state_dict()
        assert_eq(set(cd1.keys()), set(cd2.keys()))
        for k in cd1.keys():
            assert_true(th.allclose(cd1[k], cd2[k]), f"critic mismatch after load: {k}")


# =============================================================================
# Tests: OnPolicyAlgorithm integration (ACKTR)
# =============================================================================
def test_acktr_algorithm_smoke_run_runs_and_updates():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = acktr(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        hidden_sizes=(64, 64),
        rollout_steps=128,      # small for tests
        update_epochs=1,
        minibatch_size=None,    # full batch (common for ACKTR)
        gamma=0.99,
        gae_lambda=0.95,
        # KFAC knobs can be heavy; keep default but reduce steps by keeping env small
        actor_optim_name="kfac",
        critic_optim_name="kfac",
        use_amp=False,
    )

    out = _rollout_steps(algo, env, n_steps=1200, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # Metrics keys depend on your core naming; these are from your ACKTRCore implementation.
    assert_true("loss/policy" in m, "missing loss/policy")
    assert_true("loss/value" in m, "missing loss/value")
    assert_true("loss/total" in m, "missing loss/total")
    assert_true("lr/actor" in m, "missing lr/actor")
    assert_true("lr/critic" in m, "missing lr/critic")

    assert_true(np.isfinite(float(m["loss/policy"])), "loss/policy not finite")
    assert_true(np.isfinite(float(m["loss/value"])), "loss/value not finite")
    assert_true(np.isfinite(float(m["loss/total"])), "loss/total not finite")


def test_acktr_parameters_change_after_update():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = acktr(
        obs_dim=4,
        action_dim=2,
        device="cpu",
        hidden_sizes=(64, 64),
        rollout_steps=128,
        update_epochs=1,
        minibatch_size=None,
        gamma=0.99,
        gae_lambda=0.95,
        actor_optim_name="kfac",
        critic_optim_name="kfac",
        use_amp=False,
    )
    algo.setup(env)

    obs, _ = env.reset(seed=0)

    # Drive until ready_to_update() (bounded)
    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        act = algo.act(obs, deterministic=False)
        act_np = _as_float_action(act, env.act_dim)

        next_obs, r, terminated, truncated, _info = env.step(act_np)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": act_np,
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


def test_acktr_rejects_invalid_action_shape_if_enforced_by_head():
    seed_all(0)

    head = ACKTRHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # Deliberately "odd" action shapes
    bad_acts = [
        th.tensor(0.0),                 # ()
        th.zeros((2,), dtype=th.float32),      # (action_dim,)
        th.zeros((1, 2, 1), dtype=th.float32), # (B, action_dim, 1)  <-- truly weird
    ]

    enforced = False
    for bad_act in bad_acts:
        try:
            _ = head.evaluate_actions(obs, bad_act)
        except Exception:
            enforced = True
            break

    if enforced:
        # If head enforces strict shapes, it must raise for at least one bad shape
        assert_raises(Exception, lambda: head.evaluate_actions(obs, th.zeros((1, 2, 1), dtype=th.float32)))
    else:
        # If head normalizes shapes, this test should not require an exception.
        # Instead, sanity-check the output contract.
        out = head.evaluate_actions(obs, th.tensor(0.0))
        assert_true(isinstance(out, dict), "evaluate_actions must return a dict")
        assert_true("log_prob" in out and "value" in out and "entropy" in out, "missing keys in evaluate_actions output")


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("acktr_head_actor_dist_and_evaluate_shapes", test_acktr_head_actor_dist_and_evaluate_shapes),
    ("acktr_head_save_and_load_roundtrip", test_acktr_head_save_and_load_roundtrip),
    ("acktr_algorithm_smoke_run_runs_and_updates", test_acktr_algorithm_smoke_run_runs_and_updates),
    ("acktr_parameters_change_after_update", test_acktr_parameters_change_after_update),
    ("acktr_rejects_invalid_action_shape_if_enforced_by_head", test_acktr_rejects_invalid_action_shape_if_enforced_by_head),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="acktr")


if __name__ == "__main__":
    raise SystemExit(main())
