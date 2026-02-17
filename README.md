# RL Algorithms

A modular and research-oriented implementation of **model-free reinforcement learning algorithms** in PyTorch.

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
rl_algorithms/
│
├── model_free/
│ ├── baselines/
│ │ ├── policy_gradients/
│ │ │ ├── on_policy/
│ │ │ │ ├── ppo/
│ │ │ │ ├── trpo/
│ │ │ │ ├── vpg/
│ │ │ │ ├── a2c/
│ │ │ │ └── acktr/
│ │ │ │
│ │ │ └── off_policy/
│ │ │ ├── sac/
│ │ │ ├── td3/
│ │ │ ├── ddpg/
│ │ │ ├── tqc/
│ │ │ ├── redq/
│ │ │ └── acer/
│ │ │
│ │ └── value_based/
│ │ ├── dqn/
│ │ ├── qrdqn/
│ │ └── rainbow/
│ │
│ ├── common/
│ │ ├── buffers/
│ │ ├── networks/
│ │ ├── policies/
│ │ ├── trainers/
│ │ ├── callbacks/
│ │ ├── loggers/
│ │ ├── noises/
│ │ ├── optimizers/
│ │ ├── wrappers/
│ │ └── utils/
│ │
│ └── tests/
│
└── README.md


---

## Design Philosophy

This codebase follows a modular separation of responsibilities:

- **Networks (Heads)** — define policy and value networks
- **Cores** — define loss functions and update logic
- **Algorithms** — compose heads and cores into complete agents
- **Buffers** — handle experience storage
- **Trainers** — manage training loops and environment interaction

This structure allows easy experimentation and extension.

---

## Installation

Clone the repository:
git clone https://github.com/jyhong99/rl_algorithms.git
cd rl_algorithms

Set Python path:
export PYTHONPATH="$PWD:$PYTHONPATH"

Install dependencies:
pip install torch numpy

Optional dependencies:
pip install gym
pip install ray

---

## Example Usage

Example: training SAC on a continuous control environment.

import gymnasium as gym

from model_free.baselines.policy_gradients.off_policy.sac import sac
from model_free.common.trainers.trainer_builder import build_trainer

def make_env():
return gym.make("Pendulum-v1")

algo = sac(
obs_dim=3,
action_dim=1,
device="cpu",
hidden_sizes=(256, 256),
)

trainer = build_trainer(
make_train_env=make_env,
algo=algo,
total_env_steps=200000,
)

trainer.train()

---

## Applications

This library is designed to be applicable to:

- continuous control
- robotics
- simulation optimization
- analog circuit optimization
- engineering design automation
- surrogate-assisted optimization

---

## Future Work

- improved benchmarking tools
- better distributed training support
- enhanced logging and visualization
- additional model-based RL support

---

## Author

Junyoung Hong  
M.S. Student, AI Semiconductor Engineering  
Hanyang University

Research focus:

- Reinforcement learning
- Analog circuit optimization
- SPICE simulation automation
- AI-driven EDA

GitHub: https://github.com/jyhong99

---

## License

MIT License
