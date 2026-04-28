"""Equivalent-circuit model definitions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn

from eis_fit.config import ParameterBounds, ParameterRange
from eis_fit.config_runtime import parameter_bounds_dict, resolve_parameter_range


@dataclass(slots=True, frozen=True)
class CircuitParameters:
    """Concrete circuit parameters expressed in physical units."""

    rs: float
    rsei: float
    cpe1_t: float
    cpe1_p: float
    rct: float
    cpe2_t: float
    cpe2_p: float
    w1_r: float
    w1_t: float
    w1_p: float

    def to_dict(self) -> dict[str, float]:
        """Returns a plain mapping suitable for serialization."""
        return {
            "rs": self.rs,
            "rsei": self.rsei,
            "cpe1_t": self.cpe1_t,
            "cpe1_p": self.cpe1_p,
            "rct": self.rct,
            "cpe2_t": self.cpe2_t,
            "cpe2_p": self.cpe2_p,
            "w1_r": self.w1_r,
            "w1_t": self.w1_t,
            "w1_p": self.w1_p,
        }


def impedance_from_parameters(
    frequencies_hz: torch.Tensor,
    parameters: CircuitParameters,
    complex_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes the complex impedance for one parameter set.

    Args:
        frequencies_hz: One-dimensional frequency tensor.
        parameters: Circuit parameters in physical units.
        complex_dtype: Optional complex dtype override.

    Returns:
        Complex impedance values for each frequency.
    """
    resolved_complex_dtype = complex_dtype or torch.complex128
    real_dtype = (
        torch.float32 if resolved_complex_dtype == torch.complex64 else torch.float64
    )
    omega = 2.0 * math.pi * frequencies_hz.to(dtype=real_dtype)
    rs = torch.as_tensor(parameters.rs, dtype=real_dtype, device=omega.device)
    rsei = torch.as_tensor(parameters.rsei, dtype=real_dtype, device=omega.device)
    cpe1_t = torch.as_tensor(parameters.cpe1_t, dtype=real_dtype, device=omega.device)
    cpe1_p = torch.as_tensor(parameters.cpe1_p, dtype=real_dtype, device=omega.device)
    rct = torch.as_tensor(parameters.rct, dtype=real_dtype, device=omega.device)
    cpe2_t = torch.as_tensor(parameters.cpe2_t, dtype=real_dtype, device=omega.device)
    cpe2_p = torch.as_tensor(parameters.cpe2_p, dtype=real_dtype, device=omega.device)
    w1_r = torch.as_tensor(parameters.w1_r, dtype=real_dtype, device=omega.device)
    w1_t = torch.as_tensor(parameters.w1_t, dtype=real_dtype, device=omega.device)
    w1_p = torch.as_tensor(parameters.w1_p, dtype=real_dtype, device=omega.device)
    return _impedance_from_tensor_parameters(
        omega=omega,
        rs=rs,
        rsei=rsei,
        cpe1_t=cpe1_t,
        cpe1_p=cpe1_p,
        rct=rct,
        cpe2_t=cpe2_t,
        cpe2_p=cpe2_p,
        w1_r=w1_r,
        w1_t=w1_t,
        w1_p=w1_p,
        complex_dtype=resolved_complex_dtype,
    )


class EquivalentCircuitModel(nn.Module):
    """Trainable impedance model with constrained parameters."""

    def __init__(
        self,
        bounds: ParameterBounds,
        device: torch.device,
        real_dtype: torch.dtype,
        complex_dtype: torch.dtype,
        seed: int,
    ) -> None:
        """Initializes the model and randomly samples a parameter start point."""
        super().__init__()
        self._bounds_by_name = parameter_bounds_dict(bounds)
        self._device = device
        self._real_dtype = real_dtype
        self._complex_dtype = complex_dtype
        self._raw_parameters = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros((), dtype=real_dtype, device=device))
                for name in self._bounds_by_name
            }
        )
        self.initialize_random_(seed)

    def initialize_random_(self, seed: int) -> None:
        """Samples a random valid initialization."""
        rng = random.Random(seed)
        with torch.no_grad():
            for name, bounds in self._bounds_by_name.items():
                sampled_value = _sample_range(name, bounds, rng)
                raw_value = _inverse_transform(name, sampled_value, bounds)
                self._raw_parameters[name].copy_(
                    torch.tensor(raw_value, dtype=self._real_dtype, device=self._device)
                )

    def current_parameters(self) -> CircuitParameters:
        """Returns the current parameters in physical units."""
        transformed = self.parameter_tensors()
        values = {name: float(tensor.detach().cpu().item()) for name, tensor in transformed.items()}
        return CircuitParameters(**values)

    def parameter_tensors(self) -> dict[str, torch.Tensor]:
        """Returns current parameters as bounded tensors."""
        return self._transformed_parameters()

    def forward(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        """Evaluates the circuit on the provided frequencies."""
        transformed = self.parameter_tensors()
        omega = 2.0 * math.pi * frequencies_hz.to(device=self._device, dtype=self._real_dtype)
        return _impedance_from_tensor_parameters(
            omega=omega,
            rs=transformed["rs"],
            rsei=transformed["rsei"],
            cpe1_t=transformed["cpe1_t"],
            cpe1_p=transformed["cpe1_p"],
            rct=transformed["rct"],
            cpe2_t=transformed["cpe2_t"],
            cpe2_p=transformed["cpe2_p"],
            w1_r=transformed["w1_r"],
            w1_t=transformed["w1_t"],
            w1_p=transformed["w1_p"],
            complex_dtype=self._complex_dtype,
        )

    def _transformed_parameters(self) -> dict[str, torch.Tensor]:
        """Transforms raw trainable tensors into bounded physical parameters."""
        return {
            name: _transform_parameter(name, self._raw_parameters[name], bounds)
            for name, bounds in self._bounds_by_name.items()
        }


def _parallel_resistor_cpe(
    resistance: torch.Tensor,
    cpe_t: torch.Tensor,
    cpe_p: torch.Tensor,
    omega: torch.Tensor,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Computes impedance of a resistor in parallel with a constant phase element."""
    admittance = (1.0 / resistance).to(dtype=complex_dtype) + cpe_t.to(
        dtype=complex_dtype
    ) * _complex_frequency_power(omega, cpe_p, complex_dtype)
    return 1.0 / admittance


def _open_warburg(
    resistance: torch.Tensor,
    tau: torch.Tensor,
    exponent: torch.Tensor,
    omega: torch.Tensor,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Computes the finite-length open Warburg impedance."""
    frequency_term = _complex_frequency_power(omega * tau, exponent, complex_dtype)
    return resistance.to(dtype=complex_dtype) / (torch.tanh(frequency_term) * frequency_term)


def _impedance_from_tensor_parameters(
    omega: torch.Tensor,
    rs: torch.Tensor,
    rsei: torch.Tensor,
    cpe1_t: torch.Tensor,
    cpe1_p: torch.Tensor,
    rct: torch.Tensor,
    cpe2_t: torch.Tensor,
    cpe2_p: torch.Tensor,
    w1_r: torch.Tensor,
    w1_t: torch.Tensor,
    w1_p: torch.Tensor,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Computes impedance directly from tensor parameters."""
    zp1 = _parallel_resistor_cpe(rsei, cpe1_t, cpe1_p, omega, complex_dtype)
    zp2 = _parallel_resistor_cpe(rct, cpe2_t, cpe2_p, omega, complex_dtype)
    zw = _open_warburg(w1_r, w1_t, w1_p, omega, complex_dtype)
    rs_complex = torch.complex(rs, torch.zeros_like(rs)).to(dtype=complex_dtype)
    return rs_complex + zp1 + zp2 + zw


def _complex_frequency_power(
    omega_like: torch.Tensor,
    exponent: torch.Tensor,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluates ``(j * omega_like) ** exponent`` in a differentiable form."""
    complex_omega = torch.complex(
        torch.zeros_like(omega_like, dtype=omega_like.dtype),
        omega_like,
    ).to(dtype=complex_dtype)
    return torch.exp(exponent.to(dtype=complex_omega.real.dtype) * torch.log(complex_omega))


def _sample_range(name: str, bounds: ParameterRange, rng: random.Random) -> float:
    """Samples one random value from the provided bounds."""
    lower, upper, log_scale = resolve_parameter_range(name, bounds)
    if log_scale:
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        return math.exp(rng.uniform(log_lower, log_upper))
    return rng.uniform(lower, upper)


def _transform_parameter(name: str, raw_value: torch.Tensor, bounds: ParameterRange) -> torch.Tensor:
    """Maps an unconstrained scalar into the target bounds."""
    unit_value = torch.sigmoid(raw_value)
    lower, upper, log_scale = resolve_parameter_range(name, bounds)
    if log_scale:
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        return torch.exp(log_lower + unit_value * (log_upper - log_lower))
    return lower + unit_value * (upper - lower)


def _inverse_transform(name: str, value: float, bounds: ParameterRange) -> float:
    """Maps a bounded scalar back into the unconstrained optimization space."""
    epsilon = 1e-6
    lower, upper, log_scale = resolve_parameter_range(name, bounds)
    if log_scale:
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        unit_value = (math.log(value) - log_lower) / (log_upper - log_lower)
    else:
        unit_value = (value - lower) / (upper - lower)
    unit_value = min(max(unit_value, epsilon), 1.0 - epsilon)
    return math.log(unit_value / (1.0 - unit_value))
