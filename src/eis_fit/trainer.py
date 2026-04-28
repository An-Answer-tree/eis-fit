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

from eis_fit.config import FitConfig
from eis_fit.config_runtime import (
    resolve_complex_dtype,
    resolve_device,
    resolve_real_dtype,
    resolve_soft_range,
    soft_constraints_dict,
)
from eis_fit.data import ImpedanceDataset, load_impedance_dataset, trim_positive_imaginary_prefix
from eis_fit.model import CircuitParameters, EquivalentCircuitModel, impedance_from_parameters
from eis_fit.utils import write_fit_outputs


@dataclass(slots=True, frozen=True)
class LossRecord:
    """One loss sample from the optimization process."""

    restart: int
    epoch: int
    loss: float


@dataclass(slots=True, frozen=True)
class RestartSummary:
    """Best loss collected from one random restart."""

    restart: int
    best_loss: float
    epochs_run: int


@dataclass(slots=True)
class FitResult:
    """Collects the outcome of a fitting run."""

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
    """Runs the full fitting pipeline for one configuration."""
    dataset = load_impedance_dataset(config.input_path)
    fit_dataset_input = dataset
    skipped_prefix_points = 0
    if config.trim_inductive_prefix:
        fit_dataset_input, skipped_prefix_points = trim_positive_imaginary_prefix(dataset)

    result = fit_dataset(fit_dataset_input, config)
    frequencies = torch.as_tensor(
        dataset.frequencies_hz,
        dtype=resolve_real_dtype(),
        device=resolve_device(config),
    )
    fitted_impedance = impedance_from_parameters(
        frequencies,
        result.best_parameters,
        complex_dtype=resolve_complex_dtype(),
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
    device = resolve_device(config)
    real_dtype = resolve_real_dtype()
    complex_dtype = resolve_complex_dtype()
    frequencies, target_impedance = dataset.to_torch(device, real_dtype, complex_dtype)

    best_loss = math.inf
    best_restart = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_model: EquivalentCircuitModel | None = None
    loss_history: list[LossRecord] = []
    restart_summaries: list[RestartSummary] = []

    start_time = time.perf_counter()
    for restart in range(config.training.restarts):
        model = EquivalentCircuitModel(
            bounds=config.bounds,
            device=device,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            seed=config.training.seed + restart,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

        best_restart_loss = math.inf
        best_restart_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        epochs_run = 0

        for epoch in range(1, config.training.max_epochs + 1):
            epochs_run = epoch
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frequencies)
            fit_loss = _normalized_complex_mse(
                prediction,
                target_impedance,
                minimum_scale=config.training.minimum_modulus_scale,
            )
            loss = fit_loss + _chemistry_penalty(model.parameter_tensors(), config)

            if not torch.isfinite(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            loss_history.append(LossRecord(restart=restart, epoch=epoch, loss=loss_value))

            if loss_value + config.training.min_improvement < best_restart_loss:
                best_restart_loss = loss_value
                best_restart_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

        if best_restart_state is not None and config.training.lbfgs_max_epochs > 0:
            model.load_state_dict(best_restart_state)
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=config.training.lbfgs_learning_rate,
                max_iter=1,
                history_size=20,
                line_search_fn="strong_wolfe",
            )
            lbfgs_wait = 0
            for lbfgs_epoch in range(1, config.training.lbfgs_max_epochs + 1):
                epoch_number = epochs_run + lbfgs_epoch

                def closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(frequencies)
                    fit_loss = _normalized_complex_mse(
                        prediction,
                        target_impedance,
                        minimum_scale=config.training.minimum_modulus_scale,
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
                        minimum_scale=config.training.minimum_modulus_scale,
                    )
                    refined_loss = fit_loss + _chemistry_penalty(model.parameter_tensors(), config)
                loss_value = float(refined_loss.detach().cpu().item())
                loss_history.append(LossRecord(restart=restart, epoch=epoch_number, loss=loss_value))

                if loss_value + config.training.min_improvement < best_restart_loss:
                    best_restart_loss = loss_value
                    best_restart_state = copy.deepcopy(model.state_dict())
                    lbfgs_wait = 0
                else:
                    lbfgs_wait += 1

                epochs_run = epoch_number
                if lbfgs_wait >= config.training.lbfgs_patience:
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
    """Computes a scale-normalized mean squared complex residual."""
    residual = prediction - target
    scale = torch.clamp(torch.abs(target), min=minimum_scale)
    return torch.mean((torch.abs(residual) / scale) ** 2)


def _chemistry_penalty(parameters: dict[str, torch.Tensor], config: FitConfig) -> torch.Tensor:
    """Builds a soft chemistry penalty for non-unique parameter solutions."""
    reference_tensor = next(iter(parameters.values()))
    penalty = torch.zeros((), dtype=reference_tensor.dtype, device=reference_tensor.device)
    constraints = config.chemistry_constraints

    for name, soft_range in soft_constraints_dict(constraints).items():
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
    soft_range,
) -> torch.Tensor:
    """Penalizes movement outside a preferred chemistry range."""
    lower, upper, weight, log_scale = resolve_soft_range(name, soft_range)
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


def parse_config(args: Sequence[str] | None = None) -> FitConfig:
    """Parses ``FitConfig`` from the command line via draccus."""
    return draccus.parse(
        FitConfig,
        args=args,
        prog="python -m eis_fit.trainer",
        preferred_help="inline",
    )


def main(args: Sequence[str] | None = None) -> int:
    """Runs the trainer entry point and returns a process exit code."""
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
