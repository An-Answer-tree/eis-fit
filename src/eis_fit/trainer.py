"""Training loop for impedance fitting."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import draccus
import numpy as np
import torch
from tqdm.auto import tqdm

from eis_fit.config import ChemistryConstraints, FitConfig, SoftRangeConstraint
from eis_fit.data import ImpedanceDataset, load_impedance_dataset, trim_positive_imaginary_prefix
from eis_fit.model import CircuitParameters, EquivalentCircuitModel, impedance_from_parameters
from eis_fit.utils import write_fit_outputs


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


@dataclass(slots=True, frozen=True)
class LossRecord:
    """One loss sample from the optimization process.

    Attributes:
        restart: Restart index that produced the sample.
        epoch: Epoch number within the restart.
        loss: Scalar loss value.
    """

    restart: int
    epoch: int
    loss: float


@dataclass(slots=True, frozen=True)
class RestartSummary:
    """Best loss collected from one random restart.

    Attributes:
        restart: Restart index.
        best_loss: Best loss found during the restart.
        epochs_run: Number of epochs executed for the restart.
    """

    restart: int
    best_loss: float
    epochs_run: int


@dataclass(slots=True)
class FitResult:
    """Collects the outcome of a fitting run.

    Attributes:
        best_parameters: Best-fit circuit parameters.
        best_loss: Lowest loss found across all restarts.
        best_restart: Restart index that produced the best parameters.
        fitted_impedance: Complex fitted impedance values for the input points.
        loss_history: Per-epoch loss records.
        restart_summaries: Per-restart best-loss summaries.
        runtime_seconds: Optimization runtime in seconds.
        resolved_device: Torch device used for optimization.
        output_dir: Timestamped output directory, if outputs were written.
        input_point_count: Number of points loaded from the input file.
        fit_point_count: Number of points used for optimization.
        skipped_prefix_points: Number of leading points skipped before fitting.
    """

    best_parameters: CircuitParameters
    best_loss: float
    best_restart: int
    fitted_impedance: np.ndarray
    loss_history: list[LossRecord]
    restart_summaries: list[RestartSummary]
    runtime_seconds: float
    resolved_device: str
    output_dir: Path | None = None
    input_point_count: int = 0
    fit_point_count: int = 0
    skipped_prefix_points: int = 0


def run_fit(config: FitConfig) -> FitResult:
    """Runs the full fitting pipeline for one configuration.

    Args:
        config: Runtime configuration.

    Returns:
        Fitting result with parameters, traces, and output metadata.
    """
    dataset = load_impedance_dataset(config.input_path)
    fit_dataset_input = dataset
    skipped_prefix_points = 0
    if config.trim_inductive_prefix:
        fit_dataset_input, skipped_prefix_points = trim_positive_imaginary_prefix(dataset)

    result = fit_dataset(fit_dataset_input, config)
    frequencies = torch.as_tensor(
        dataset.frequencies_hz,
        dtype=_resolve_real_dtype(),
        device=_resolve_device(config),
    )
    fitted_impedance = impedance_from_parameters(
        frequencies,
        result.best_parameters,
        complex_dtype=_resolve_complex_dtype(),
    )
    result.fitted_impedance = fitted_impedance.detach().cpu().numpy()
    result.input_point_count = dataset.size
    result.fit_point_count = fit_dataset_input.size
    result.skipped_prefix_points = skipped_prefix_points
    write_fit_outputs(dataset, result, config)
    return result


def fit_dataset(dataset: ImpedanceDataset, config: FitConfig) -> FitResult:
    """Fits the equivalent circuit to one dataset.

    Args:
        dataset: Measured impedance spectrum.
        config: Runtime configuration.

    Returns:
        A ``FitResult`` containing best-fit parameters and traces.

    Raises:
        RuntimeError: If every restart fails.
    """
    device = _resolve_device(config)
    real_dtype = _resolve_real_dtype()
    complex_dtype = _resolve_complex_dtype()
    frequencies, target_impedance = dataset.to_torch(device, real_dtype, complex_dtype)

    best_loss = math.inf
    best_restart = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_model: EquivalentCircuitModel | None = None
    loss_history: list[LossRecord] = []
    restart_summaries: list[RestartSummary] = []

    start_time = time.perf_counter()
    restart_progress = tqdm(
        range(config.restarts),
        desc="Fitting restarts",
        unit="restart",
    )
    for restart in restart_progress:
        model = EquivalentCircuitModel(
            bounds=config.bounds,
            device=device,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            seed=config.seed + restart,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        best_restart_loss = math.inf
        best_restart_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        epochs_run = 0

        adam_progress = tqdm(
            range(1, config.max_epochs + 1),
            desc=f"Adam restart {restart + 1}/{config.restarts}",
            leave=False,
            unit="epoch",
        )
        for epoch in adam_progress:
            epochs_run = epoch
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frequencies)
            fit_loss = _normalized_complex_mse(
                prediction,
                target_impedance,
                minimum_scale=config.minimum_modulus_scale,
            )
            loss = fit_loss + _chemistry_penalty(model.parameter_tensors(), config)

            if not torch.isfinite(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            loss_history.append(LossRecord(restart=restart, epoch=epoch, loss=loss_value))

            if loss_value + config.min_improvement < best_restart_loss:
                best_restart_loss = loss_value
                best_restart_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            adam_progress.set_postfix(
                loss=f"{loss_value:.3e}",
                best=f"{best_restart_loss:.3e}",
                wait=epochs_without_improvement,
            )

            if epochs_without_improvement >= config.early_stopping_patience:
                break

        if best_restart_state is not None and config.lbfgs_max_epochs > 0:
            model.load_state_dict(best_restart_state)
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=config.lbfgs_learning_rate,
                max_iter=1,
                history_size=20,
                line_search_fn="strong_wolfe",
            )
            lbfgs_wait = 0
            lbfgs_progress = tqdm(
                range(1, config.lbfgs_max_epochs + 1),
                desc=f"LBFGS restart {restart + 1}/{config.restarts}",
                leave=False,
                unit="epoch",
            )
            for lbfgs_epoch in lbfgs_progress:
                epoch_number = epochs_run + lbfgs_epoch

                def closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(frequencies)
                    fit_loss = _normalized_complex_mse(
                        prediction,
                        target_impedance,
                        minimum_scale=config.minimum_modulus_scale,
                    )
                    loss = fit_loss + _chemistry_penalty(model.parameter_tensors(), config)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                with torch.no_grad():
                    refined_prediction = model(frequencies)
                    fit_loss = _normalized_complex_mse(
                        refined_prediction,
                        target_impedance,
                        minimum_scale=config.minimum_modulus_scale,
                    )
                    refined_loss = fit_loss + _chemistry_penalty(model.parameter_tensors(), config)
                loss_value = float(refined_loss.detach().cpu().item())
                loss_history.append(LossRecord(restart=restart, epoch=epoch_number, loss=loss_value))

                if loss_value + config.min_improvement < best_restart_loss:
                    best_restart_loss = loss_value
                    best_restart_state = copy.deepcopy(model.state_dict())
                    lbfgs_wait = 0
                else:
                    lbfgs_wait += 1

                epochs_run = epoch_number
                lbfgs_progress.set_postfix(
                    loss=f"{loss_value:.3e}",
                    best=f"{best_restart_loss:.3e}",
                    wait=lbfgs_wait,
                )
                if lbfgs_wait >= config.lbfgs_patience:
                    break

        restart_summaries.append(
            RestartSummary(restart=restart, best_loss=best_restart_loss, epochs_run=epochs_run)
        )

        if best_restart_state is None:
            continue

        if best_restart_loss < best_loss:
            best_loss = best_restart_loss
            best_restart = restart
            best_state = best_restart_state
            best_model = model
        restart_progress.set_postfix(
            best_loss=f"{best_loss:.3e}",
            best_restart=best_restart,
        )

    runtime_seconds = time.perf_counter() - start_time
    if best_state is None or best_model is None or best_restart < 0:
        raise RuntimeError("Model training did not converge to a valid solution.")

    best_model.load_state_dict(best_state)
    fitted_impedance = best_model(frequencies).detach().cpu().numpy()
    return FitResult(
        best_parameters=best_model.current_parameters(),
        best_loss=best_loss,
        best_restart=best_restart,
        fitted_impedance=fitted_impedance,
        loss_history=loss_history,
        restart_summaries=restart_summaries,
        runtime_seconds=runtime_seconds,
        resolved_device=str(device),
    )


def _normalized_complex_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_scale: float,
) -> torch.Tensor:
    """Computes a scale-normalized mean squared complex residual.

    Args:
        prediction: Predicted complex impedance values.
        target: Measured complex impedance values.
        minimum_scale: Lower bound used when normalizing residuals by modulus.

    Returns:
        Mean squared normalized complex residual.
    """
    residual = prediction - target
    scale = torch.clamp(torch.abs(target), min=minimum_scale)
    return torch.mean((torch.abs(residual) / scale) ** 2)


def _chemistry_penalty(parameters: dict[str, torch.Tensor], config: FitConfig) -> torch.Tensor:
    """Builds a soft chemistry penalty for non-unique parameter solutions."""
    reference_tensor = next(iter(parameters.values()))
    penalty = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
    constraints = config.chemistry_constraints

    for name, soft_range in _soft_constraints_dict(constraints).items():
        penalty = penalty + _soft_range_penalty(name, parameters[name], soft_range)

    if constraints.minimum_rct_to_rsei_ratio > 0.0 and constraints.rct_rsei_ratio_weight > 0.0:
        ratio = parameters["rct"] / parameters["rsei"]
        gap = torch.relu(constraints.minimum_rct_to_rsei_ratio - ratio)
        penalty = penalty + constraints.rct_rsei_ratio_weight * gap.square()

    if constraints.minimum_second_arc_tau_ratio > 0.0 and constraints.second_arc_tau_ratio_weight > 0.0:
        first_arc_tau = (parameters["rsei"] * parameters["cpe1_t"]).pow(1.0 / parameters["cpe1_p"])
        second_arc_tau = (parameters["rct"] * parameters["cpe2_t"]).pow(1.0 / parameters["cpe2_p"])
        tau_ratio = second_arc_tau / first_arc_tau
        tau_gap = torch.relu(constraints.minimum_second_arc_tau_ratio - tau_ratio)
        penalty = penalty + constraints.second_arc_tau_ratio_weight * tau_gap.square()

    return penalty


def _soft_range_penalty(
    name: str,
    parameter: torch.Tensor,
    soft_range: SoftRangeConstraint,
) -> torch.Tensor:
    """Penalizes movement outside a preferred chemistry range."""
    lower, upper, weight, log_scale = _resolve_soft_range(name, soft_range)
    if weight == 0.0:
        return torch.zeros((), dtype=parameter.dtype, device=parameter.device)

    if log_scale:
        value = torch.log(parameter)
        lower = math.log(lower)
        upper = math.log(upper)
    else:
        value = parameter

    below = torch.relu(lower - value)
    above = torch.relu(value - upper)
    return weight * (below.square() + above.square())


def _soft_constraints_dict(
    constraints: ChemistryConstraints,
) -> dict[str, SoftRangeConstraint]:
    """Returns enabled soft constraints by parameter name.

    Args:
        constraints: Chemistry-informed soft constraint configuration.

    Returns:
        Mapping of parameter names to enabled soft range tuples.
    """
    return {
        name: getattr(constraints, name)
        for name in _PARAMETER_NAMES
        if getattr(constraints, name) is not None
    }


def _resolve_soft_range(
    name: str,
    value: SoftRangeConstraint,
) -> tuple[float, float, float, bool]:
    """Returns normalized soft-constraint data for one parameter.

    Args:
        name: Parameter name used to choose the default scale.
        value: Soft range tuple in ``(lower, upper)``,
            ``(lower, upper, weight)``, or
            ``(lower, upper, weight, log_scale)`` form.

    Returns:
        A tuple of ``(lower, upper, weight, log_scale)``.
    """
    lower = float(value[0])
    upper = float(value[1])
    weight = float(value[2]) if len(value) > 2 else 1.0
    log_scale = bool(value[3]) if len(value) > 3 else name not in _LINEAR_SCALE_PARAMETERS
    return lower, upper, weight, log_scale


def _resolve_device(config: FitConfig) -> torch.device:
    """Returns the torch device that should be used for training.

    Args:
        config: Runtime configuration.

    Returns:
        Torch device selected from the configuration.

    Raises:
        RuntimeError: If CUDA is explicitly requested but unavailable.
    """
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


def _resolve_real_dtype() -> torch.dtype:
    """Returns the fixed real dtype used for optimization."""
    return torch.float64


def _resolve_complex_dtype() -> torch.dtype:
    """Returns the fixed complex dtype paired with the real dtype."""
    return torch.complex128


def parse_config(args: Sequence[str] | None = None) -> FitConfig:
    """Parses ``FitConfig`` from the command line via draccus.

    Args:
        args: Optional argument list. Uses process arguments when omitted.

    Returns:
        Parsed fit configuration.
    """
    return draccus.parse(
        FitConfig,
        args=args,
        prog="python -m eis_fit.trainer",
        preferred_help="inline",
    )


def main(args: Sequence[str] | None = None) -> int:
    """Runs the trainer entry point.

    Args:
        args: Optional argument list. Uses process arguments when omitted.

    Returns:
        Process exit code.
    """
    config = parse_config(args)
    result = run_fit(config)
    print(f"Best loss: {result.best_loss:.6e}")
    print(f"Best restart: {result.best_restart}")
    print(f"Output directory: {result.output_dir}")
    print("Best-fit parameters:")
    for name, value in result.best_parameters.to_dict().items():
        print(f"  {name}: {value:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
