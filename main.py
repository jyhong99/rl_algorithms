import numpy as np

# Prefer gymnasium; fallback to gym if needed
try:
    import gymnasium as gym
except Exception:
    import gym

from model_free.common.trainers.trainer_builder import build_trainer
from model_free.baselines.policy_gradients.off_policy.sac.sac import sac
from model_free.baselines.policy_gradients.off_policy.td3.td3 import td3
from model_free.baselines.policy_gradients.off_policy.ddpg.ddpg import ddpg
from model_free.baselines.policy_gradients.on_policy.ppo.ppo import ppo
# -----------------------------
# Env factories
# -----------------------------
def make_pendulum_env(seed: int, *, render_mode=None):
    """
    Factory helper: create a fresh Pendulum env with a given seed.
    """
    env = gym.make("Pendulum-v1", render_mode=render_mode)

    # Gymnasium reset seeding
    try:
        env.reset(seed=seed)
    except TypeError:
        # Older gym compatibility
        env.seed(seed)

    # Also seed action space for reproducibility (best-effort)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    return env


def make_train_env():
    # Use a fixed seed for reproducibility; change or randomize as desired
    return make_pendulum_env(seed=0)


def make_eval_env():
    # Different seed for evaluation
    return make_pendulum_env(seed=10)


# -----------------------------
# Infer dims from a probe env
# -----------------------------
_probe_env = make_train_env()
obs_dim = int(np.prod(_probe_env.observation_space.shape))
action_dim = int(np.prod(_probe_env.action_space.shape))
_probe_env.close()

device = "cuda"  # or "cpu"


# -----------------------------
# Build algo + trainer
# -----------------------------
algo = sac(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)

trainer = build_trainer(
    make_train_env=make_train_env,
    make_eval_env=make_eval_env,
    algo=algo,
    n_envs=5,
)

trainer.train()
