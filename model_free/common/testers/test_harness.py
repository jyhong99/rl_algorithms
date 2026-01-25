from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping
import os
import tempfile
import numpy as np
import shutil
import torch as th
import torch.nn as nn
from model_free.common.testers.test_utils import (
    assert_true,
)


from model_free.common.policies.base_core import BaseCore, ActorCriticCore, QLearningCore
from model_free.common.policies.base_head import (
    BaseHead,
    OnPolicyContinuousActorCriticHead,
    OffPolicyContinuousActorCriticHead,
    DeterministicActorCriticHead,
    QLearningHead,
)
from model_free.common.policies.base_policy import BaseAlgorithm, BasePolicyAlgorithm


@dataclass
class LoggedRecord:
    step: int
    prefix: str
    metrics: Dict[str, Any]


class FakeLogger:
    """
    Minimal logger capturing log(...) payloads for assertions.
    Matches your log_utils.log(trainer, payload, step=..., prefix=...) pattern
    if your log() ultimately calls trainer.logger.log(...).
    """
    def __init__(self) -> None:
        self.records: List[LoggedRecord] = []

    def log(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        self.records.append(LoggedRecord(step=int(step), prefix=str(prefix), metrics=dict(metrics)))


class FakeCallbacks:
    """Optional aggregator if you have trainer.callbacks.on_eval_end(...) calls."""
    def __init__(self, callbacks: List[Any]) -> None:
        self._callbacks = callbacks

    def on_eval_end(self, trainer: Any, metrics: Dict[str, Any]) -> bool:
        ok = True
        for cb in self._callbacks:
            fn = getattr(cb, "on_eval_end", None)
            if callable(fn):
                ok = bool(fn(trainer, metrics)) and ok
        return ok

    def on_checkpoint(self, trainer: Any, path: str) -> bool:
        ok = True
        for cb in self._callbacks:
            fn = getattr(cb, "on_checkpoint", None)
            if callable(fn):
                ok = bool(fn(trainer, path)) and ok
        return ok


class FakeAlgo:
    """Algo stub for LRLoggingCallback / GradParamNormCallback."""
    def __init__(self) -> None:
        self.optimizers = {}
        self.schedulers = {}

    def get_lr_dict(self) -> Dict[str, float]:
        return {}  # override per-test if needed


class FakeTrainer:
    """
    Duck-typed trainer with knobs/counters used by your callbacks.
    NOTE: In your code, safe_int_attr(trainer) is used for step and also for upd in some callbacks.
    For robust tests, expose explicit fields and ensure safe_int_attr reads the right one.
    """
    def __init__(self) -> None:
        self.logger = FakeLogger()

        # Counters (set these in tests)
        self.global_env_step: int = 0
        self.global_update_step: int = 0

        # Some callbacks use safe_int_attr(trainer) ambiguously; keep a single 'step' too
        self.step: int = 0
        self.update_step: int = 0

        # Checkpoint plumbing
        self.ckpt_dir = tempfile.mkdtemp(prefix="ckpt_test_")
        self.checkpoint_prefix = "ckpt_"

        self._saved: List[str] = []  # saved checkpoint paths

        # Eval plumbing
        self._next_eval_metrics: Optional[Dict[str, Any]] = None

        # Algorithm
        self.algo = FakeAlgo()

        # Callback aggregator if needed by EvalCallback(dispatch_eval_end=True)
        self.callbacks: Optional[FakeCallbacks] = None

    def set_eval_metrics(self, m: Dict[str, Any]) -> None:
        self._next_eval_metrics = dict(m)

    # ---- contract methods ----
    def run_evaluation(self) -> Optional[Dict[str, Any]]:
        return self._next_eval_metrics

    def save_checkpoint(self, path: Optional[str] = None) -> Optional[str]:
        """
        Make a real file so CheckpointCallback rotation/glob works.
        If path is provided and looks like a filename, save there; else auto name under ckpt_dir.
        """
        if path is None or path == "":
            fname = f"{self.checkpoint_prefix}{self.step:012d}.pt"
            p = os.path.join(self.ckpt_dir, fname)
        else:
            # if given relative, resolve into ckpt_dir to keep tests isolated
            p = path if os.path.isabs(path) else os.path.join(self.ckpt_dir, path)

        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"test")
        self._saved.append(p)
        return p
    

class DummyBoxSpace:
    def __init__(self, low, high, shape):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = tuple(shape)

    def sample(self):
        return np.random.uniform(self.low, self.high, size=self.shape).astype(np.float32)


class DummyEnvGym4:
    """step -> (obs, reward, done, info)"""
    def __init__(self, obs_shape=(3,), action_shape=(2,)):
        self.observation_space = DummyBoxSpace(low=-np.ones(obs_shape), high=np.ones(obs_shape), shape=obs_shape)
        self.action_space = DummyBoxSpace(low=-2*np.ones(action_shape), high=2*np.ones(action_shape), shape=action_shape)
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        self._t += 1
        obs = np.full(self.observation_space.shape, self._t, dtype=np.float32)
        reward = 1.0
        done = (self._t >= 3)
        info = {}
        return obs, reward, done, info


class DummyEnvGymnasium5:
    """step -> (obs, reward, terminated, truncated, info), reset -> (obs, info)"""
    def __init__(self, obs_shape=(3,), action_shape=(2,)):
        self.observation_space = DummyBoxSpace(low=-np.ones(obs_shape), high=np.ones(obs_shape), shape=obs_shape)
        self.action_space = DummyBoxSpace(low=-2*np.ones(action_shape), high=2*np.ones(action_shape), shape=action_shape)
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = np.full(self.observation_space.shape, self._t, dtype=np.float32)
        reward = 1.0
        terminated = (self._t >= 3)
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


# =============================================================================
# Dummy Algo for Trainer integration tests
# =============================================================================
class DummyAlgo:
    """
    Minimal algorithm that satisfies Trainer contract.
    - Always returns zeros action
    - Increments env step counter via on_env_step
    - Becomes ready_to_update() after warmup_steps
    - update() returns metrics with sys/num_updates=1
    """

    is_off_policy: bool = True

    def __init__(self, *, action_shape: Tuple[int, ...], warmup_steps: int = 3) -> None:
        self.action_shape = tuple(action_shape)
        self.warmup_steps = int(warmup_steps)
        self._env_steps = 0
        self._num_updates = 0
        self._training = False

    def setup(self, env: Any) -> None:
        # no-op
        return

    def set_training(self, training: bool) -> None:
        self._training = bool(training)

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        return np.zeros(self.action_shape, dtype=np.float32)

    def on_env_step(self, transition: Dict[str, Any]) -> None:
        self._env_steps += 1

    def ready_to_update(self) -> bool:
        return self._env_steps >= self.warmup_steps

    def update(self) -> Mapping[str, Any]:
        self._num_updates += 1
        return {"sys/num_updates": 1, "dummy/loss": float(self._num_updates)}

    # checkpoint hooks expected by save_checkpoint/load_checkpoint
    def save(self, path: str) -> None:
        # minimal artifact: numpy save
        np.save(path + ".npy", np.array([self._env_steps, self._num_updates], dtype=np.int64))

    def load(self, path: str) -> None:
        arr = np.load(path + ".npy")
        self._env_steps = int(arr[0])
        self._num_updates = int(arr[1])



# =============================================================================
# Helper models
# =============================================================================
class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 8, hid: int = 16, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=True),
            nn.Tanh(),
            nn.Linear(hid, out_dim, bias=True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU()
        self.head = nn.Linear(out_ch * 8 * 8, 5, bias=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        z = self.act(self.conv(x))
        z = z.view(z.size(0), -1)
        return self.head(z)


# =============================================================================
# Helper: temporary directory fixture
# =============================================================================
class TempDir:
    def __init__(self) -> None:
        self.path = tempfile.mkdtemp(prefix="writers_test_")

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            shutil.rmtree(self.path, ignore_errors=True)
        except Exception:
            pass


# =============================================================================
# In-memory writer stub for Logger testing
# =============================================================================
class MemoryWriter:
    def __init__(self) -> None:
        self.rows: List[Dict[str, float]] = []
        self.flush_calls = 0
        self.close_calls = 0

    def write(self, row: Dict[str, float]) -> None:
        self.rows.append(dict(row))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.close_calls += 1


# =============================================================================
# Callback runner (same behavior as your tests)
# =============================================================================
class CallbackRunner:
    """
    Simple lifecycle driver:
      - on_train_start
      - repeated on_step / on_update
      - optional on_eval_end dispatch
    """
    def __init__(self, trainer: FakeTrainer, callbacks: List[Any]) -> None:
        self.trainer = trainer
        self.callbacks = callbacks
        self.trainer.callbacks = FakeCallbacks(callbacks)

    def train_start(self) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, "on_train_start", None)
            if callable(fn):
                ok = fn(self.trainer)
                assert_true(ok is True, f"{type(cb).__name__}.on_train_start returned {ok!r}")

    def step(self, transition: Optional[Dict[str, Any]] = None) -> bool:
        self.trainer.step = self.trainer.global_env_step

        for cb in self.callbacks:
            fn = getattr(cb, "on_step", None)
            if callable(fn):
                ok = fn(self.trainer, transition=transition)
                if ok is False:
                    return False
        return True

    def update(self, metrics: Optional[Dict[str, Any]] = None) -> bool:
        self.trainer.update_step = self.trainer.global_update_step

        for cb in self.callbacks:
            fn = getattr(cb, "on_update", None)
            if callable(fn):
                ok = fn(self.trainer, metrics=metrics)
                if ok is False:
                    return False
        return True

    def eval_end(self, metrics: Dict[str, Any]) -> bool:
        for cb in self.callbacks:
            fn = getattr(cb, "on_eval_end", None)
            if callable(fn):
                ok = fn(self.trainer, metrics)
                if ok is False:
                    return False
        return True


# =============================================================================
# Minimal test doubles
# =============================================================================
class TinyActor(nn.Module):
    """Actor with act() and get_dist() interface (simple Gaussian)."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, act_dim))
        self.log_std = nn.Parameter(th.zeros(act_dim))

    def act(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, Dict[str, Any]]:
        mu = self.net(obs)
        if deterministic:
            return mu, {}
        std = self.log_std.exp().expand_as(mu)
        return mu + std * th.randn_like(mu), {}

    def get_dist(self, obs: th.Tensor):
        mu = self.net(obs)
        std = self.log_std.exp().expand_as(mu)
        return th.distributions.Normal(mu, std)


class TinyCritic(nn.Module):
    def __init__(self, obs_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class TinyQ(nn.Module):
    def __init__(self, obs_dim: int = 4, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.ReLU(), nn.Linear(16, n_actions))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class DummyHeadForCore:
    """Duck-typed head for BaseCore tests; supplies target update helpers."""
    def __init__(self, device: str = "cpu") -> None:
        self.device = th.device(device)

    def freeze_target(self, module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    @th.no_grad()
    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        target.load_state_dict(source.state_dict())

    @th.no_grad()
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float = 1.0) -> None:
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.data.mul_(1.0 - tau).add_(ps.data, alpha=tau)


class DummyCore(BaseCore):
    """Concrete core for BaseCore tests."""
    def __init__(self, *, head: Any, use_amp: bool = False) -> None:
        super().__init__(head=head, use_amp=use_amp)

    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        self._bump()
        return {"loss": 1.0}


class DummyACHead(OnPolicyContinuousActorCriticHead):
    """Concrete OnPolicyActorCriticHead with minimal actor/critic attached."""
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.actor = TinyActor(obs_dim=4, act_dim=2).to(self.device)
        self.critic = TinyCritic(obs_dim=4).to(self.device)


class DummyQLHead(QLearningHead):
    def __init__(self, *, device: str = "cpu", n_actions: int = 5) -> None:
        super().__init__(device=device)
        self.n_actions = int(n_actions)
        self.q = TinyQ(obs_dim=4, n_actions=n_actions).to(self.device)
        self.q_target = TinyQ(obs_dim=4, n_actions=n_actions).to(self.device)


# =============================================================================
# Fake env & buffers (to avoid gym dependency in tests)
# =============================================================================
class FakeSpace:
    def __init__(self, shape: Tuple[int, ...], *, n: Optional[int] = None, low: Any = None, high: Any = None) -> None:
        self.shape = shape
        self.n = n
        if low is not None:
            self.low = np.asarray(low, dtype=np.float32)
        if high is not None:
            self.high = np.asarray(high, dtype=np.float32)

    def sample(self):
        if self.n is not None:
            return int(np.random.randint(0, self.n))
        if hasattr(self, "low") and hasattr(self, "high"):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(np.float32)
        return np.random.randn(*self.shape).astype(np.float32)


class FakeEnv:
    def __init__(self, *, obs_dim: int = 4, act_dim: int = 2) -> None:
        self.observation_space = FakeSpace((obs_dim,))
        self.action_space = FakeSpace((act_dim,), low=-1.0, high=1.0)
        self.action_rescale = True  # to test warmup policy-action-space sampling


class FakeReplayBuffer:
    def __init__(self) -> None:
        self.size = 0

    def add(self, **kwargs: Any) -> None:
        self.size += 1

    def sample(self, batch_size: int, **kwargs: Any):
        # minimal batch object; core.update_from_batch may accept any
        return {"batch_size": int(batch_size)}


class FakeRolloutBuffer:
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = int(buffer_size)
        self.full = False
        self.pos = 0

    def add(self, **kwargs: Any) -> None:
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_value: float, last_done: bool) -> None:
        # no-op for this test
        return

    def sample(self, batch_size: int, shuffle: bool = False):
        # Yield exactly ceil(pos/batch_size) minibatches for determinism
        n = self.pos
        bs = int(batch_size)
        i = 0
        while i < n:
            j = min(n, i + bs)
            yield {"i": i, "j": j}
            i = j

    def reset(self) -> None:
        self.pos = 0
        self.full = False

# Tests: BaseAlgorithm save/load (with dummy head/core that expose state)
# =============================================================================
class StatefulHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(th.tensor([1.0]))

    def set_training(self, training: bool) -> None:
        self.train(training)

    def act(self, obs: Any, deterministic: bool = False) -> th.Tensor:
        return th.tensor([0])

class StatefulCore:
    def __init__(self) -> None:
        self.k = 0

    def state_dict(self) -> Dict[str, Any]:
        return {"k": self.k}

    def load_state_dict(self, s: Mapping[str, Any]) -> None:
        self.k = int(s.get("k", 0))


class AlgoForSaveLoad(BaseAlgorithm):
    pass


# =============================================================================
# Tests: OffPolicyAlgorithm scheduling + sys/num_updates accounting
# =============================================================================
class DummyOffPolicyCore:
    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        # Return scalar metrics only
        return {"loss": 1.0}

class DummyOffPolicyHead:
    def __init__(self) -> None:
        self.device = th.device("cpu")
        self._noise_reset = 0

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        return np.zeros((2,), dtype=np.float32)

    def reset_exploration_noise(self) -> None:
        self._noise_reset += 1

# =============================================================================
# Additional test doubles for ActorCriticCore / QLearningCore
# =============================================================================
class TinyStateActionCritic(nn.Module):
    """critic(s,a)->q with simple MLP"""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, s: th.Tensor, a: th.Tensor) -> th.Tensor:
        x = th.cat([s, a], dim=-1)
        return self.net(x)


class DummyDeterministicActor(nn.Module):
    """actor.act(obs)->action"""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, act_dim))

    def act(self, obs: th.Tensor, deterministic: bool = True):
        a = self.net(obs)
        return a, {}


class ACHeadForCore(DummyHeadForCore):
    """head providing actor/critic for ActorCriticCore"""
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.actor = DummyDeterministicActor(4, 2).to(self.device)
        self.critic = TinyStateActionCritic(4, 2).to(self.device)


class QHeadForCore(DummyHeadForCore):
    def __init__(self, device: str = "cpu", n_actions: int = 5) -> None:
        super().__init__(device=device)
        self.q = TinyQ(obs_dim=4, n_actions=n_actions).to(self.device)


class ConcreteActorCriticCore(ActorCriticCore):
    """
    Minimal concrete implementation:
    - critic MSE loss on dummy target
    - actor maximize Q (deterministic policy gradient style)
    """
    def __init__(self, *, head: Any, **kwargs: Any) -> None:
        super().__init__(head=head, **kwargs)

    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        self.head.actor.train(True)
        self.head.critic.train(True)

        # fake batch
        s = th.randn(32, 4, device=self.device)
        a = th.randn(32, 2, device=self.device)
        y = th.randn(32, 1, device=self.device)

        # critic update
        q = self.head.critic(s, a)
        critic_loss = th.mean((q - y) ** 2)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_params(self.head.critic.parameters(), max_grad_norm=1.0, optimizer=self.critic_opt)
        self.critic_opt.step()

        # actor update: maximize Q(s, actor(s))
        a_pi, _ = self.head.actor.act(s, deterministic=True)
        actor_loss = -self.head.critic(s, a_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_params(self.head.actor.parameters(), max_grad_norm=1.0, optimizer=self.actor_opt)
        self.actor_opt.step()

        # sched step (if any)
        self._step_scheds()
        self._bump()

        return {"loss/critic": float(critic_loss.detach().cpu().item()),
                "loss/actor": float(actor_loss.detach().cpu().item())}


class ConcreteQLearningCore(QLearningCore):
    """Minimal Q-learning update: MSE to dummy target on selected actions."""
    def __init__(self, *, head: Any, n_actions: int = 5, **kwargs: Any) -> None:
        self.n_actions = int(n_actions)
        super().__init__(head=head, **kwargs)

    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        self.head.q.train(True)

        s = th.randn(32, 4, device=self.device)
        a_idx = th.randint(0, self.n_actions, (32,), device=self.device)
        y = th.randn(32, device=self.device)

        q_all = self.head.q(s)                 # (B,A)
        q_sa = q_all.gather(1, a_idx.view(-1, 1)).squeeze(1)
        loss = th.mean((q_sa - y) ** 2)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_params(self.head.q.parameters(), max_grad_norm=1.0, optimizer=self.opt)
        self.opt.step()
        self._step_sched()
        self._bump()

        return {"loss/q": float(loss.detach().cpu().item())}


class SACLikeActor(nn.Module):
    """Actor with get_dist() and act() for OffPolicyActorCriticHead."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, act_dim))
        self.log_std = nn.Parameter(th.zeros(act_dim))

    def get_dist(self, obs: th.Tensor):
        mu = self.net(obs)
        std = self.log_std.exp().expand_as(mu)
        return th.distributions.Normal(mu, std)

    def act(self, obs: th.Tensor, deterministic: bool = False):
        dist = self.get_dist(obs)
        if deterministic:
            return dist.mean, {}
        return dist.rsample(), {}


class SACLikeCritic(nn.Module):
    """Return two Q estimates to match your OffPolicyActorCriticHead.q_values signature."""
    def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
        super().__init__()
        self.q1 = TinyStateActionCritic(obs_dim, act_dim)
        self.q2 = TinyStateActionCritic(obs_dim, act_dim)

    def forward(self, s: th.Tensor, a: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1(s, a), self.q2(s, a)


class DummyOffPolicyACHead(OffPolicyContinuousActorCriticHead):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.actor = SACLikeActor(4, 2).to(self.device)
        self.critic = SACLikeCritic(4, 2).to(self.device)
        self.critic_target = SACLikeCritic(4, 2).to(self.device)

class DummyDetHead(DeterministicActorCriticHead):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.actor = DummyDeterministicActor(4, 2).to(self.device)
        self.critic = TinyStateActionCritic(4, 2).to(self.device)
        self.actor_target = DummyDeterministicActor(4, 2).to(self.device)
        self.critic_target = TinyStateActionCritic(4, 2).to(self.device)
        self.action_low = np.array([-0.5, -0.25], np.float32)
        self.action_high = np.array([0.5, 0.25], np.float32)
        self.noise = None

# =============================================================================
# BasePolicyAlgorithm test
# =============================================================================
class MinimalPolicyAlgo(BasePolicyAlgorithm):
    def __init__(self, *, head: Any, core: Any, device: str = "cpu") -> None:
        super().__init__(head=head, core=core, device=device)
        self._ready = False

    def setup(self, env: Any) -> None:
        self._ready = False

    def on_env_step(self, transition: Dict[str, Any]) -> None:
        # emulate BasePolicyAlgorithm semantics: env_steps increments in subclass
        self._env_steps += 1
        # become ready after 3 steps
        if self._env_steps >= 3:
            self._ready = True

    def ready_to_update(self) -> bool:
        return bool(self._ready)

    def update(self) -> Dict[str, float]:
        # just return a metric and reset readiness
        self._ready = False
        return {"dummy/update": 1.0}

# =============================================================================
# Additional head tests: BaseHead / OffPolicyActorCriticHead / DeterministicActorCriticHead
# =============================================================================
class ConcreteBaseHead(BaseHead):
    pass


# =============================================================================
# Tests: OnPolicyAlgorithm sys/num_updates == num_minibatches
# =============================================================================
class DummyOnPolicyCore:
    def __init__(self):
        self.calls = 0

    def update_from_batch(self, batch: Any) -> Dict[str, Any]:
        self.calls += 1
        return {"loss": 1.0}

class DummyOnPolicyHead:
    def __init__(self):
        self.device = th.device("cpu")

    def set_training(self, training: bool) -> None:
        return

    def act(self, obs: Any, deterministic: bool = False) -> Any:
        return np.zeros((2,), dtype=np.float32)

    def evaluate_actions(self, obs: Any, action: Any) -> Mapping[str, Any]:
        # value and log_prob must be scalar-like for OnPolicyAlgorithm
        return {"value": 0.0, "log_prob": 0.0}

    def value_only(self, obs: Any) -> Any:
        return 0.0