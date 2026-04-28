"""Runtime helpers for configuration values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from eis_fit.config import ChemistryConstraints, FitConfig, ParameterBounds, ParameterRange


_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_LINEAR_SCALE_PARAMETERS = frozenset({"cpe1_p", "cpe2_p", "w1_p"})
_PARAMETER_NAMES = (
    "rs",
    "rsei",
    "cpe1_t",
    "cpe1_p",
    "rct",
    "cpe2_t",
    "cpe2_p",
    "w1_r",
    "w1_t",
    "w1_p",
)


def parameter_bounds_dict(bounds: ParameterBounds) -> dict[str, ParameterRange]:
    """Returns the configured hard bounds by parameter name."""
    return {name: getattr(bounds, name) for name in _PARAMETER_NAMES}


def soft_constraints_dict(constraints: ChemistryConstraints) -> dict[str, tuple[Any, ...]]:
    """Returns the enabled soft ranges by parameter name."""
    return {
        name: getattr(constraints, name)
        for name in _PARAMETER_NAMES
        if getattr(constraints, name) is not None
    }


def resolve_parameter_range(name: str, value: Sequence[object]) -> tuple[float, float, bool]:
    """Returns ``(lower, upper, log_scale)`` for one hard bound."""
    lower = float(value[0])
    upper = float(value[1])
    log_scale = bool(value[2]) if len(value) > 2 else name not in _LINEAR_SCALE_PARAMETERS
    return lower, upper, log_scale


def resolve_soft_range(name: str, value: Sequence[object]) -> tuple[float, float, float, bool]:
    """Returns ``(lower, upper, weight, log_scale)`` for one soft bound."""
    lower = float(value[0])
    upper = float(value[1])
    weight = float(value[2]) if len(value) > 2 else 1.0
    log_scale = bool(value[3]) if len(value) > 3 else name not in _LINEAR_SCALE_PARAMETERS
    return lower, upper, weight, log_scale


def resolve_device(config: FitConfig) -> torch.device:
    """Returns the torch device that should be used for training."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use device='cpu' only if you intend "
            "to override the GPU-first default."
        )
    return torch.device(config.device)


def resolve_real_dtype() -> torch.dtype:
    """Returns the fixed real dtype used for optimization."""
    return torch.float64


def resolve_complex_dtype() -> torch.dtype:
    """Returns the fixed complex dtype paired with the real dtype."""
    return torch.complex128


def resolve_output_dir(config: FitConfig) -> Path:
    """Builds a timestamped output directory path for one run."""
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    return Path(config.output_root) / timestamp


def config_to_dict(config: FitConfig) -> dict[str, Any]:
    """Serializes config data into JSON-safe values."""
    return _serialize(asdict(config))


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
