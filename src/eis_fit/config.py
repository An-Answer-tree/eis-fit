"""Static configuration values for impedance fitting.

This module intentionally only defines dataclass-based configuration. Runtime
helpers, validation, serialization, and derived values live in the modules that
use them.
"""

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
    """Hard bounds for trainable equivalent-circuit parameters.

    Each field accepts either ``(lower, upper)`` or
    ``(lower, upper, log_scale)``. When ``log_scale`` is omitted, positive
    magnitude-like parameters are optimized on a logarithmic scale and exponent
    parameters are optimized on a linear scale.

    Attributes:
        rs: Bounds for the series resistance.
        rsei: Bounds for the SEI-film resistance.
        cpe1_t: Bounds for the first constant-phase element coefficient.
        cpe1_p: Bounds for the first constant-phase element exponent.
        rct: Bounds for the charge-transfer resistance.
        cpe2_t: Bounds for the second constant-phase element coefficient.
        cpe2_p: Bounds for the second constant-phase element exponent.
        w1_r: Bounds for the finite-length Warburg resistance.
        w1_t: Bounds for the finite-length Warburg time constant.
        w1_p: Bounds for the finite-length Warburg exponent.
    """

    rs: ParameterRange = (0.05, 50.0)
    rsei: ParameterRange = (0.1, 1000.0)
    cpe1_t: ParameterRange = (1e-10, 1.0)
    cpe1_p: ParameterRange = (0.4, 1.0)
    rct: ParameterRange = (0.1, 3000.0)
    cpe2_t: ParameterRange = (1e-10, 1.0)
    cpe2_p: ParameterRange = (0.4, 1.0)
    w1_r: ParameterRange = (0.1, 3000.0)
    w1_t: ParameterRange = (1e-6, 1000.0)
    w1_p: ParameterRange = (0.25, 1.0)


@dataclass(slots=True, frozen=True)
class ChemistryConstraints:
    """Soft chemistry-informed constraints for non-unique fits.

    Parameter range fields accept ``None`` to disable the constraint, or one of
    ``(lower, upper)``, ``(lower, upper, weight)``, or
    ``(lower, upper, weight, log_scale)``. Values outside a soft range remain
    allowed, but increase the training loss according to ``weight``.

    Attributes:
        rs: Preferred range for the series resistance.
        rsei: Preferred range for the SEI-film resistance.
        cpe1_t: Preferred range for the first CPE coefficient.
        cpe1_p: Preferred range for the first CPE exponent.
        rct: Preferred range for the charge-transfer resistance.
        cpe2_t: Preferred range for the second CPE coefficient.
        cpe2_p: Preferred range for the second CPE exponent.
        w1_r: Preferred range for the Warburg resistance.
        w1_t: Preferred range for the Warburg time constant.
        w1_p: Preferred range for the Warburg exponent.
        minimum_rct_to_rsei_ratio: Minimum preferred ``Rct / Rsei`` ratio.
        rct_rsei_ratio_weight: Loss weight for violating the ``Rct / Rsei``
            ratio.
        minimum_second_arc_tau_ratio: Minimum preferred second-to-first arc time
            constant ratio.
        second_arc_tau_ratio_weight: Loss weight for violating the arc time
            constant ratio.
    """

    rs: SoftRangeConstraint | None = (1.0, 25.0, 0.02)
    rsei: SoftRangeConstraint | None = (1.0, 300.0, 0.01)
    cpe1_t: SoftRangeConstraint | None = None
    cpe1_p: SoftRangeConstraint | None = (0.55, 1.0, 0.02, False)
    rct: SoftRangeConstraint | None = (10.0, 1500.0, 0.01)
    cpe2_t: SoftRangeConstraint | None = None
    cpe2_p: SoftRangeConstraint | None = (0.55, 1.0, 0.02, False)
    w1_r: SoftRangeConstraint | None = (1.0, 1500.0, 0.005)
    w1_t: SoftRangeConstraint | None = None
    w1_p: SoftRangeConstraint | None = (0.35, 0.75, 0.02, False)
    minimum_rct_to_rsei_ratio: float = 1.0
    rct_rsei_ratio_weight: float = 0.1
    minimum_second_arc_tau_ratio: float = 2.0
    second_arc_tau_ratio_weight: float = 0.1


@dataclass(slots=True)
class FitConfig:
    """Top-level configuration for one impedance fitting run.

    Attributes:
        input_path: ZView/Z60W-style text file to fit.
        output_root: Root directory for timestamped output folders.
        device: Torch device name. Use ``"cuda"`` for GPU, ``"cpu"`` for CPU,
            or ``"auto"`` to prefer CUDA when available.
        generate_plots: Whether to write Nyquist and Bode plot images.
        trim_inductive_prefix: Whether to drop leading positive-imaginary points
            before optimization.
        restarts: Number of random initializations to try.
        max_epochs: Maximum Adam epochs for each restart.
        learning_rate: Adam optimizer learning rate.
        lbfgs_learning_rate: LBFGS refinement learning rate.
        lbfgs_max_epochs: Maximum LBFGS refinement epochs for each restart.
        lbfgs_patience: LBFGS early-stopping patience.
        early_stopping_patience: Adam early-stopping patience.
        min_improvement: Minimum loss decrease required to reset patience.
        gradient_clip_norm: Maximum gradient norm before optimizer updates.
        seed: Base random seed; each restart adds its index to this value.
        minimum_modulus_scale: Lower bound for target impedance normalization.
        bounds: Hard parameter bounds.
        chemistry_constraints: Soft chemistry-informed constraints.
    """

    input_path: Path = Path("data/EA-2.txt")
    output_root: Path = Path("outputs")
    device: str = "cpu"
    generate_plots: bool = True
    trim_inductive_prefix: bool = True
    restarts: int = 20
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
    bounds: ParameterBounds = ParameterBounds()
    chemistry_constraints: ChemistryConstraints = ChemistryConstraints()
