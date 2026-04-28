"""Configuration values for impedance fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ParameterRange = tuple[float, float] | tuple[float, float, bool]
SoftRangeConstraint = (
    tuple[float, float]
    | tuple[float, float, float]
    | tuple[float, float, float, bool]
)


@dataclass(slots=True, frozen=True)
class ParameterBounds:
    rs: ParameterRange = (1e-3, 1e4)
    rsei: ParameterRange = (1e-3, 1e6)
    cpe1_t: ParameterRange = (1e-10, 1.0)
    cpe1_p: ParameterRange = (0.3, 1.2)
    rct: ParameterRange = (1e-3, 1e6)
    cpe2_t: ParameterRange = (1e-10, 1.0)
    cpe2_p: ParameterRange = (0.3, 1.2)
    w1_r: ParameterRange = (1e-3, 1e6)
    w1_t: ParameterRange = (1e-6, 1e4)
    w1_p: ParameterRange = (0.2, 1.0)


@dataclass(slots=True, frozen=True)
class ChemistryConstraints:
    rs: SoftRangeConstraint | None = None
    rsei: SoftRangeConstraint | None = None
    cpe1_t: SoftRangeConstraint | None = None
    cpe1_p: SoftRangeConstraint | None = None
    rct: SoftRangeConstraint | None = None
    cpe2_t: SoftRangeConstraint | None = None
    cpe2_p: SoftRangeConstraint | None = None
    w1_r: SoftRangeConstraint | None = None
    w1_t: SoftRangeConstraint | None = None
    w1_p: SoftRangeConstraint | None = None
    minimum_rct_to_rsei_ratio: float = 0.0
    rct_rsei_ratio_weight: float = 0.0
    minimum_second_arc_tau_ratio: float = 0.0
    second_arc_tau_ratio_weight: float = 0.0


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    restarts: int = 12
    max_epochs: int = 3000
    learning_rate: float = 0.03
    lbfgs_learning_rate: float = 0.5
    lbfgs_max_epochs: int = 250
    lbfgs_patience: int = 40
    early_stopping_patience: int = 400
    min_improvement: float = 1e-7
    gradient_clip_norm: float = 5.0
    seed: int = 12345
    minimum_modulus_scale: float = 1.0


@dataclass(slots=True)
class FitConfig:
    input_path: Path
    output_root: Path = Path("outputs")
    device: str = "cuda"
    generate_plots: bool = True
    trim_inductive_prefix: bool = True
    training: TrainingConfig = TrainingConfig()
    bounds: ParameterBounds = ParameterBounds()
    chemistry_constraints: ChemistryConstraints = ChemistryConstraints()
