from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th


# =============================================================================
# Local import bootstrap (no installation required)
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

# IMPORTANT:
# Adjust these imports to your actual package layout if needed.
from model_free.baselines.policy_gradients.on_policy.a2c.a2c import a2c  # noqa: E402
from model_free.baselines.policy_gradients.on_policy.a2c.head import A2CHead  # noqa: E402


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
class DummyContinuousEnv:
    """
    Tiny episodic continuous-control env.

    Shapes
    ------
    - obs: (obs_dim,) float32
    - action: (act_dim,) float32 in [-1, 1] (we clip)

    Dynamics
    --------
    obs <- clip(obs + 0.1 * action + noise, -1, 1)

    Episode ends after `horizon` steps.
    Reward = 1.0 per step (smoke-testing only).
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

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        a = np.clip(a, -1.0, 1.0)

        noise = self._rng.randn(self.obs_dim).astype(np.float32) * 0.01
        self._obs = np.clip(self._obs + 0.1 * a.mean() + noise, -1.0, 1.0).astype(np.float32)

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


def _make_small_algo(*, device: str = "cpu") -> Any:
    # Continuous A2C for DummyContinuousEnv(obs_dim=4, act_dim=2)
    return a2c(
        obs_dim=4,
        action_dim=2,
        device=device,
        hidden_sizes=(64, 64),
        rollout_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=1,
        minibatch_size=None,  # full-batch for stability in tests
        vf_coef=0.5,
        ent_coef=0.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        max_grad_norm=10.0,
        use_amp=False,
        normalize_advantages=False,
    )


def _rollout_steps(algo: Any, env: Any, *, n_steps: int, seed: int = 0, do_updates: bool = True) -> Dict[str, Any]:
    """
    Generic rollout driver for your OnPolicyAlgorithm-style interface.

    Assumptions (matching your OffPolicy tests style)
    -------------------------------------------------
    - algo.setup(env)
    - algo.act(obs, deterministic=False) -> action
    - algo.on_env_step(transition_dict)
    - algo.ready_to_update() -> bool
    - algo.update() -> metrics dict
    """
    obs, _ = env.reset(seed=seed)
    algo.setup(env)

    num_updates = 0
    last_metrics: Any = None

    for _t in range(int(n_steps)):
        a = algo.act(obs, deterministic=False)

        next_obs, r, terminated, truncated, _info = env.step(a)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a,
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
# Tests: A2CHead (continuous)
# =============================================================================
def test_a2c_head_act_and_evaluate_shapes():
    seed_all(0)

    head = A2CHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        log_std_mode="param",
        log_std_init=-0.5,
        device="cpu",
    )

    obs = np.zeros((4,), dtype=np.float32)

    # act() should return (B, act_dim) where B=1
    a = head.act(obs, deterministic=False)
    assert_true(th.is_tensor(a), "head.act should return torch.Tensor")
    assert_eq(tuple(a.shape), (1, 2))

    # evaluate_actions should return tensors with batch dimension B=1
    out = head.evaluate_actions(obs, a, as_scalar=False)
    assert_true(isinstance(out, dict), "evaluate_actions must return dict")
    assert_true("value" in out and "log_prob" in out and "entropy" in out, "missing keys from evaluate_actions")

    v = out["value"]
    lp = out["log_prob"]
    ent = out["entropy"]

    assert_true(th.is_tensor(v) and th.is_tensor(lp) and th.is_tensor(ent), "outputs must be torch tensors")
    assert_eq(tuple(v.shape), (1, 1))
    # log_prob/entropy are standardized by your head to (B,1)
    assert_eq(tuple(lp.shape), (1, 1))
    assert_eq(tuple(ent.shape), (1, 1))


def test_a2c_head_save_and_load_roundtrip():
    seed_all(0)

    head1 = A2CHead(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        activation_fn=th.nn.ReLU,
        init_type="orthogonal",
        gain=1.0,
        bias=0.0,
        log_std_mode="param",
        log_std_init=-0.5,
        device="cpu",
    )

    _ = head1.act(np.zeros((4,), dtype=np.float32))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a2c_head_ckpt")
        head1.save(path)

        head2 = A2CHead(
            obs_dim=4,
            action_dim=2,
            hidden_sizes=(32, 32),
            activation_fn=th.nn.ReLU,
            init_type="orthogonal",
            gain=1.0,
            bias=0.0,
            log_std_mode="param",
            log_std_init=-0.5,
            device="cpu",
        )
        head2.load(path)

        # actor round-trip
        sd1 = head1.actor.state_dict()
        sd2 = head2.actor.state_dict()
        assert_eq(set(sd1.keys()), set(sd2.keys()))
        for k in sd1.keys():
            assert_true(th.allclose(sd1[k], sd2[k]), f"actor mismatch after load: {k}")

        # critic round-trip
        sd1c = head1.critic.state_dict()
        sd2c = head2.critic.state_dict()
        assert_eq(set(sd1c.keys()), set(sd2c.keys()))
        for k in sd1c.keys():
            assert_true(th.allclose(sd1c[k], sd2c[k]), f"critic mismatch after load: {k}")


# =============================================================================
# Tests: OnPolicyAlgorithm integration (smoke)
# =============================================================================
def test_a2c_smoke_run_runs_and_updates():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu")
    out = _rollout_steps(algo, env, n_steps=2000, seed=0, do_updates=True)
    env.close()

    assert_true(out["num_updates"] > 0, "expected at least one update to run")
    m = out["last_metrics"]
    assert_true(isinstance(m, dict) and len(m) > 0, "update() should return non-empty metrics")

    # These keys are generic expectations. Adjust if your logger prefixes differ.
    assert_true(("loss/policy" in m) or ("loss/actor" in m), "missing policy/actor loss key")
    assert_true(("loss/value" in m) or ("loss/critic" in m), "missing value/critic loss key")
    assert_true(any(k.startswith("lr/") for k in m.keys()), "expected some lr/* metric")

    # Basic sanity: finite losses
    for k in ["loss/policy", "loss/value", "loss/total", "loss/actor", "loss/critic"]:
        if k in m:
            assert_true(np.isfinite(float(m[k])), f"{k} not finite")


def test_a2c_parameters_change_after_update():
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=4, act_dim=2, horizon=50, seed=0)

    algo = _make_small_algo(device="cpu")
    algo.setup(env)

    # Drive until ready_to_update
    obs, _ = env.reset(seed=0)

    max_drive = 5000
    drove = 0
    while drove < max_drive and (not algo.ready_to_update()):
        a = algo.act(obs, deterministic=False)
        next_obs, r, terminated, truncated, _info = env.step(a)
        done = bool(terminated or truncated)

        algo.on_env_step(
            {
                "obs": obs,
                "action": a,
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


def test_a2c_on_env_step_missing_fields_raises_if_enforced():
    """
    Optional contract test.

    If your OnPolicyAlgorithm enforces transition dict keys,
    this should raise. If it does not enforce, you can remove this test.
    """
    seed_all(0)
    env = DummyContinuousEnv(obs_dim=4, act_dim=2, horizon=10, seed=0)

    algo = _make_small_algo(device="cpu")
    algo.setup(env)

    obs, _ = env.reset(seed=0)
    a = algo.act(obs, deterministic=False)
    next_obs, r, terminated, truncated, _info = env.step(a)
    done = bool(terminated or truncated)

    transition_missing = {
        "obs": obs,
        "action": a,
        # "reward": float(r),
        "next_obs": next_obs,
        "done": done,
    }

    # If your algo doesn't validate, this will fail; in that case delete this test.
    assert_raises(Exception, lambda: algo.on_env_step(transition_missing))
    env.close()


# =============================================================================
# Runner
# =============================================================================
TESTS: List[Tuple[str, Callable[[], Any]]] = [
    ("a2c_head_act_and_evaluate_shapes", test_a2c_head_act_and_evaluate_shapes),
    ("a2c_head_save_and_load_roundtrip", test_a2c_head_save_and_load_roundtrip),
    ("a2c_smoke_run_runs_and_updates", test_a2c_smoke_run_runs_and_updates),
    ("a2c_parameters_change_after_update", test_a2c_parameters_change_after_update),
    # Remove this if your OnPolicyAlgorithm does not enforce transition keys.
    ("a2c_on_env_step_missing_fields_raises_if_enforced", test_a2c_on_env_step_missing_fields_raises_if_enforced),
]


def main(argv=None) -> int:
    return run_tests(TESTS, argv=argv, suite_name="a2c")


if __name__ == "__main__":
    raise SystemExit(main())
