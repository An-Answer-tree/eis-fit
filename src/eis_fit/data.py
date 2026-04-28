"""Data loading utilities for impedance spectra."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class DataFormatError(ValueError):
    """Raised when an input file does not match the expected format."""


@dataclass(slots=True, frozen=True)
class ImpedanceDataset:
    """Holds one impedance spectrum.

    Attributes:
        frequencies_hz: Frequency values in hertz.
        z_real_ohm: Measured real impedance values.
        z_imag_ohm: Measured imaginary impedance values.
        source_path: Optional path to the source file.
    """

    frequencies_hz: np.ndarray
    z_real_ohm: np.ndarray
    z_imag_ohm: np.ndarray
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalizes arrays and validates basic shape constraints."""
        frequencies_hz = np.asarray(self.frequencies_hz, dtype=np.float64)
        z_real_ohm = np.asarray(self.z_real_ohm, dtype=np.float64)
        z_imag_ohm = np.asarray(self.z_imag_ohm, dtype=np.float64)

        if frequencies_hz.ndim != 1:
            raise DataFormatError("frequencies_hz must be one-dimensional.")
        if len(frequencies_hz) == 0:
            raise DataFormatError("At least one impedance point is required.")
        if len(frequencies_hz) != len(z_real_ohm) or len(frequencies_hz) != len(z_imag_ohm):
            raise DataFormatError("Frequency, real, and imaginary arrays must have equal length.")
        if np.any(frequencies_hz <= 0.0):
            raise DataFormatError("All frequency values must be strictly positive.")
        if not np.isfinite(frequencies_hz).all():
            raise DataFormatError("Frequency values must all be finite.")
        if not np.isfinite(z_real_ohm).all() or not np.isfinite(z_imag_ohm).all():
            raise DataFormatError("Impedance values must all be finite.")

        object.__setattr__(self, "frequencies_hz", frequencies_hz)
        object.__setattr__(self, "z_real_ohm", z_real_ohm)
        object.__setattr__(self, "z_imag_ohm", z_imag_ohm)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def size(self) -> int:
        """Returns the number of samples."""
        return int(self.frequencies_hz.shape[0])

    @property
    def complex_impedance(self) -> np.ndarray:
        """Returns the complex impedance array."""
        return self.z_real_ohm + 1j * self.z_imag_ohm

    def to_torch(
        self,
        device: torch.device,
        real_dtype: torch.dtype,
        complex_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the dataset into torch tensors.

        Args:
            device: Target device.
            real_dtype: Real dtype used for frequencies.
            complex_dtype: Complex dtype used for impedance values.

        Returns:
            A tuple of frequency tensor and complex impedance tensor.
        """
        frequencies = torch.as_tensor(self.frequencies_hz, dtype=real_dtype, device=device)
        impedance = torch.as_tensor(self.complex_impedance, dtype=complex_dtype, device=device)
        return frequencies, impedance


def load_impedance_dataset(path: str | Path) -> ImpedanceDataset:
    """Loads a ZView/Z60W-style text file.

    The parser expects numeric rows with at least nine comma-separated columns and
    extracts frequency, real impedance, and imaginary impedance from columns
    ``0``, ``4``, and ``5``.

    Args:
        path: File to load.

    Returns:
        An ``ImpedanceDataset`` instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataFormatError: If no valid numeric rows are found.
    """
    source_path = Path(path)
    rows: list[tuple[float, float, float]] = []

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                frequency = float(row[0].strip())
                z_real = float(row[4].strip())
                z_imag = float(row[5].strip())
            except ValueError:
                continue
            rows.append((frequency, z_real, z_imag))

    if not rows:
        raise DataFormatError(f"No impedance rows were found in {source_path}.")

    frequencies, z_real, z_imag = (np.asarray(values, dtype=np.float64) for values in zip(*rows))
    return ImpedanceDataset(
        frequencies_hz=frequencies,
        z_real_ohm=z_real,
        z_imag_ohm=z_imag,
        source_path=source_path,
    )


def trim_positive_imaginary_prefix(
    dataset: ImpedanceDataset,
) -> tuple[ImpedanceDataset, int]:
    """Drops the leading positive-imaginary prefix from a dataset.

    This is useful when the chosen equivalent circuit does not include an
    inductive element and therefore cannot reproduce a high-frequency positive
    imaginary loop.

    Args:
        dataset: Dataset to trim.

    Returns:
        A tuple of ``(trimmed_dataset, dropped_count)``.

    Raises:
        DataFormatError: If every point has positive imaginary impedance.
    """
    first_non_positive_index = 0
    for index, value in enumerate(dataset.z_imag_ohm):
        if value <= 0.0:
            first_non_positive_index = index
            break
    else:
        raise DataFormatError(
            "All points have positive imaginary impedance; the configured circuit "
            "cannot be fitted without an inductive element."
        )

    if first_non_positive_index == 0:
        return dataset, 0

    trimmed_dataset = ImpedanceDataset(
        frequencies_hz=dataset.frequencies_hz[first_non_positive_index:],
        z_real_ohm=dataset.z_real_ohm[first_non_positive_index:],
        z_imag_ohm=dataset.z_imag_ohm[first_non_positive_index:],
        source_path=dataset.source_path,
    )
    return trimmed_dataset, first_non_positive_index
