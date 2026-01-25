"""
callbacks subpackage
--------------------

Callback API + standard callbacks + builder utilities.

NOTE
----
Some parts of the codebase reference `build_callbacks` via `callback_utils`,
but the actual implementation lives in `callback_builder.py`.
To prevent ImportError and keep call-sites stable, we re-export the correct symbol.
"""

from .base_callback import BaseCallback, CallbackList

from .eval_callback import EvalCallback
from .checkpoint_callback import CheckpointCallback
from .best_model_callback import BestModelCallback
from .early_stop_callback import EarlyStopCallback
from .nan_guard_callback import NaNGuardCallback

# ---- optional / extra callbacks (may be absent in minimal deployments) ----
try:  # pragma: no cover
    from .config_and_env_info_callback import ConfigAndEnvInfoCallback
except Exception:  # pragma: no cover
    ConfigAndEnvInfoCallback = None  # type: ignore

try:  # pragma: no cover
    from .episode_stats_callback import EpisodeStatsCallback
except Exception:  # pragma: no cover
    EpisodeStatsCallback = None  # type: ignore

try:  # pragma: no cover
    from .timing_callback import TimingCallback
except Exception:  # pragma: no cover
    TimingCallback = None  # type: ignore

try:  # pragma: no cover
    from .lr_logging_callback import LRLoggingCallback
except Exception:  # pragma: no cover
    LRLoggingCallback = None  # type: ignore

try:  # pragma: no cover
    from .grad_param_norm_callback import GradParamNormCallback
except Exception:  # pragma: no cover
    GradParamNormCallback = None  # type: ignore

# ---- optional Ray callbacks ----
try:  # pragma: no cover
    from .ray_report_callback import RayReportCallback
    from .ray_tune_checkpoint_callback import RayTuneCheckpointCallback
except Exception:  # pragma: no cover
    RayReportCallback = None  # type: ignore
    RayTuneCheckpointCallback = None  # type: ignore


# Builder (fixed import path)
from .callback_builder import build_callbacks


__all__ = [
    "BaseCallback",
    "CallbackList",
    "EvalCallback",
    "CheckpointCallback",
    "BestModelCallback",
    "EarlyStopCallback",
    "NaNGuardCallback",
    "ConfigAndEnvInfoCallback",
    "EpisodeStatsCallback",
    "TimingCallback",
    "LRLoggingCallback",
    "GradParamNormCallback",
    "RayReportCallback",
    "RayTuneCheckpointCallback",
    "build_callbacks",
]