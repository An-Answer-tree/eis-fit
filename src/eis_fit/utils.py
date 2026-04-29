"""Utilities for exporting and plotting fit results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eis_fit.config import FitConfig
from eis_fit.data import ImpedanceDataset

if TYPE_CHECKING:
    from eis_fit.trainer import FitResult


_BEST_PARAMS_FILENAME = "best_params.json"
_FIT_CURVE_FILENAME = "fit_curve.csv"
_LOSS_HISTORY_FILENAME = "loss_history.csv"
_RESTART_SUMMARY_FILENAME = "restart_summary.csv"
_NYQUIST_FILENAME = "nyquist.png"
_BODE_FILENAME = "bode.png"
_FIGURE_DPI = 160
_NYQUIST_FIGSIZE = (7.0, 5.0)
_BODE_FIGSIZE = (7.0, 7.0)
_MEASURED_LABEL = "Measured"
_FITTED_LABEL = "Fitted"
_MARKER_SIZE = 4.0
_LINE_WIDTH = 2.0
_NYQUIST_TITLE = "Nyquist Plot"
_BODE_TITLE = "Bode Plot"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def write_fit_outputs(dataset: ImpedanceDataset, result: FitResult, config: FitConfig) -> Path:
    """Writes the standard output artifacts for one fit run.

    Args:
        dataset: Original measured impedance dataset.
        result: Fitting result to serialize.
        config: Runtime configuration used for the fit.

    Returns:
        Timestamped output directory.
    """
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    result.output_dir = output_dir

    _write_best_params(output_dir / _BEST_PARAMS_FILENAME, result, config)
    _write_fit_curve(output_dir / _FIT_CURVE_FILENAME, dataset, result)
    _write_loss_history(output_dir / _LOSS_HISTORY_FILENAME, result)
    _write_restart_summary(output_dir / _RESTART_SUMMARY_FILENAME, result)

    if config.generate_plots:
        write_fit_plots(output_dir, dataset, result.fitted_impedance)

    return output_dir


def write_fit_plots(
    output_dir: Path,
    dataset: ImpedanceDataset,
    fitted_impedance: np.ndarray,
) -> None:
    """Writes Nyquist and Bode plots for one fit run.

    Args:
        output_dir: Directory that receives plot files.
        dataset: Original measured impedance dataset.
        fitted_impedance: Complex fitted impedance values.
    """
    _write_nyquist_plot(output_dir / _NYQUIST_FILENAME, dataset, fitted_impedance)
    _write_bode_plot(output_dir / _BODE_FILENAME, dataset, fitted_impedance)


def _write_best_params(path: Path, result: FitResult, config: FitConfig) -> None:
    """Serializes the best-fit parameters and run metadata."""
    payload = {
        "best_loss": result.best_loss,
        "best_restart": result.best_restart,
        "runtime_seconds": result.runtime_seconds,
        "resolved_device": result.resolved_device,
        "input_point_count": result.input_point_count,
        "fit_point_count": result.fit_point_count,
        "skipped_prefix_points": result.skipped_prefix_points,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        "parameters": result.best_parameters.to_dict(),
        "config": _config_to_dict(config),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_fit_curve(path: Path, dataset: ImpedanceDataset, result: FitResult) -> None:
    """Writes measured and fitted impedance values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "z_real_exp_ohm",
                "z_imag_exp_ohm",
                "z_real_fit_ohm",
                "z_imag_fit_ohm",
                "used_for_fit",
            ]
        )
        for index, (frequency, z_real_exp, z_imag_exp, z_fit) in enumerate(
            zip(
                dataset.frequencies_hz,
                dataset.z_real_ohm,
                dataset.z_imag_ohm,
                result.fitted_impedance,
                strict=True,
            )
        ):
            writer.writerow(
                [
                    frequency,
                    z_real_exp,
                    z_imag_exp,
                    z_fit.real,
                    z_fit.imag,
                    index >= result.skipped_prefix_points,
                ]
            )


def _write_loss_history(path: Path, result: FitResult) -> None:
    """Writes per-epoch loss values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["restart", "epoch", "loss"])
        for record in result.loss_history:
            writer.writerow([record.restart, record.epoch, record.loss])


def _write_restart_summary(path: Path, result: FitResult) -> None:
    """Writes one summary row for each random restart."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["restart", "best_loss", "epochs_run"])
        for summary in result.restart_summaries:
            writer.writerow([summary.restart, summary.best_loss, summary.epochs_run])


def _write_nyquist_plot(
    output_path: Path,
    dataset: ImpedanceDataset,
    fitted_impedance: np.ndarray,
) -> None:
    """Creates a Nyquist plot comparing measured and fitted spectra."""
    figure, axis = plt.subplots(figsize=_NYQUIST_FIGSIZE)
    axis.plot(
        dataset.z_real_ohm,
        -dataset.z_imag_ohm,
        "o",
        label=_MEASURED_LABEL,
        markersize=_MARKER_SIZE,
    )
    axis.plot(
        fitted_impedance.real,
        -fitted_impedance.imag,
        "-",
        label=_FITTED_LABEL,
        linewidth=_LINE_WIDTH,
    )
    axis.set_xlabel(r"$Z^\prime$ (Ohm)")
    axis.set_ylabel(r"$-Z^{\prime\prime}$ (Ohm)")
    axis.set_title(_NYQUIST_TITLE)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def _resolve_output_dir(config: FitConfig) -> Path:
    """Builds a timestamped output directory path for one run.

    Args:
        config: Runtime configuration.

    Returns:
        Timestamped output directory path under ``config.output_root``.
    """
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    return Path(config.output_root) / timestamp


def _config_to_dict(config: FitConfig) -> dict[str, Any]:
    """Serializes config data into JSON-safe values.

    Args:
        config: Runtime configuration.

    Returns:
        JSON-safe dictionary representation.
    """
    return _serialize(asdict(config))


def _serialize(value: Any) -> Any:
    """Recursively converts dataclass values into JSON-safe primitives.

    Args:
        value: Value to serialize.

    Returns:
        JSON-safe value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _write_bode_plot(
    output_path: Path,
    dataset: ImpedanceDataset,
    fitted_impedance: np.ndarray,
) -> None:
    """Creates a Bode magnitude and phase plot."""
    figure, axes = plt.subplots(2, 1, figsize=_BODE_FIGSIZE, sharex=True)

    measured_magnitude = np.abs(dataset.complex_impedance)
    fitted_magnitude = np.abs(fitted_impedance)
    measured_phase = np.degrees(np.angle(dataset.complex_impedance))
    fitted_phase = np.degrees(np.angle(fitted_impedance))

    axes[0].loglog(
        dataset.frequencies_hz,
        measured_magnitude,
        "o",
        label=_MEASURED_LABEL,
        markersize=_MARKER_SIZE,
    )
    axes[0].loglog(
        dataset.frequencies_hz,
        fitted_magnitude,
        "-",
        label=_FITTED_LABEL,
        linewidth=_LINE_WIDTH,
    )
    axes[0].set_ylabel("|Z| (Ohm)")
    axes[0].set_title(_BODE_TITLE)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].semilogx(
        dataset.frequencies_hz,
        measured_phase,
        "o",
        label=_MEASURED_LABEL,
        markersize=_MARKER_SIZE,
    )
    axes[1].semilogx(
        dataset.frequencies_hz,
        fitted_phase,
        "-",
        label=_FITTED_LABEL,
        linewidth=_LINE_WIDTH,
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)
