# RL Algorithms

A modular and research-oriented implementation of model-free reinforcement learning algorithms in PyTorch.

This repository provides clean, reusable, and extensible implementations of modern RL algorithms with a consistent architecture, designed for:

- reinforcement learning research
- algorithm prototyping
- benchmarking and experimentation
- integration into real-world optimization systems (e.g., circuit optimization, control)

---

## Features

- Unified and modular design across algorithms
- Support for both on-policy and off-policy methods
- Continuous and discrete action spaces
- Reusable training pipeline
- Scalable training support (optional Ray integration)
- Designed for research clarity and extensibility

---

## Implemented Algorithms

### Policy Gradient (On-Policy)

- PPO (Proximal Policy Optimization)
- TRPO (Trust Region Policy Optimization)
- VPG (Vanilla Policy Gradient)
- A2C (Advantage Actor-Critic)
- ACKTR (Kronecker-Factored Trust Region)

Discrete variants available for:

- PPO (Discrete)
- VPG (Discrete)
- A2C (Discrete)

---

### Actor-Critic (Off-Policy)

- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- DDPG (Deep Deterministic Policy Gradient)
- TQC (Truncated Quantile Critics)
- REDQ (Randomized Ensemble Double Q-learning)
- ACER (Actor-Critic with Experience Replay)

Discrete variant available for:

- SAC (Discrete)

---

### Value-Based

- DQN (Deep Q-Network)
- QR-DQN (Quantile Regression DQN)
- Rainbow DQN

---

## Repository Structure

```
rl_algorithms/
│
├── model_free/
│   ├── baselines/
│   │   ├── policy_gradients/
│   │   │   ├── on_policy/
│   │   │   │   ├── ppo/
│   │   │   │   ├── trpo/
│   │   │   │   ├── vpg/
│   │   │   │   ├── a2c/
│   │   │   │   └── acktr/
│   │   │   │
│   │   │   └── off_policy/
│   │   │       ├── sac/
│   │   │       ├── td3/
│   │   │       ├── ddpg/
│   │   │       ├── tqc/
│   │   │       ├── redq/
│   │   │       └── acer/
│   │   │
│   │   └── value_based/
│   │       ├── dqn/
│   │       ├── qrdqn/
│   │       └── rainbow/
│   │
│   ├── common/
│   │   ├── buffers/
│   │   ├── networks/
│   │   ├── policies/
│   │   ├── trainers/
│   │   ├── callbacks/
│   │   ├── loggers/
│   │   ├── noises/
│   │   ├── optimizers/
│   │   ├── wrappers/
│   │   └── utils/
│   │
│   └── tests/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jyhong99/rl_algorithms.git
cd rl_algorithms
```

Set Python path:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Install dependencies:

```bash
pip install torch numpy
```

Optional:

```bash
pip install gymnasium
pip install ray
```

---

## Example Usage

```python
import gymnasium as gym

from model_free.baselines.policy_gradients.off_policy.sac import sac
from model_free.common.trainers.trainer_builder import build_trainer

def make_env():
    return gym.make("Pendulum-v1")

algo = sac(
    obs_dim=3,
    action_dim=1,
)

trainer = build_trainer(
    make_train_env=make_env,
    algo=algo,
    total_env_steps=200000,
)

trainer.train()
```

---

## Applications

- continuous control
- robotics
- circuit optimization
- engineering optimization
- EDA systems

---

## Author

Junyoung Hong  
M.S. Student, AI Semiconductor Engineering  
Hanyang University  

GitHub: https://github.com/jyhong99

---

## License

MIT License
