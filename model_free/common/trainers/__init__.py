"""
trainers subpackage
-------------------

Training loop drivers and builders.

NOTE
----
Some call-sites reference `build_trainer` from `train_utils`, but the actual
builder lives in `trainer_builder.py`. We re-export the correct symbol.
"""

from .trainer import Trainer
from .evaluator import Evaluator

# Builder (fixed import path)
from .trainer_builder import build_trainer

__all__ = [
    "Trainer",
    "Evaluator",
    "build_trainer",
]
