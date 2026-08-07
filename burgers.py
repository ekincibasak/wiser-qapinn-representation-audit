from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from torch import nn

try:
    from .derivatives import PINNDerivatives, compute_pinn_derivatives
except ImportError:
    from derivatives import PINNDerivatives, compute_pinn_derivatives


@dataclass
class BurgersEquationResult:
    """Prediction, derivatives, and residual for viscous Burgers' equation."""

    u: torch.Tensor
    u_x: torch.Tensor
    u_t: torch.Tensor
    u_xx: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class BurgersReferenceSolution:
    """Cole-Hopf reference solution on a Cartesian space-time grid."""

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    viscosity: float
    quadrature_order: int
    reference_method: str


def burgers_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Return the standard initial condition u(x,0) = -sin(pi*x)."""
    return -torch.sin(math.pi * x)


def burgers_equation_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    viscosity: float,
) -> BurgersEquationResult:
    """Calculate the viscous Burgers residual.

    The PDE is

        u_t + u*u_x = viscosity*u_xx,

    so the residual is

        r_theta = u_t + u_theta*u_x - viscosity*u_xx.
    """
    _validate_viscosity(viscosity)

    derivatives: PINNDerivatives = compute_pinn_derivatives(
        model=model,
        x=x,
        t=t,
    )

    residual = (
        derivatives.u_t
        + derivatives.u * derivatives.u_x
        - viscosity * derivatives.u_xx
    )

    return BurgersEquationResult(
        u=derivatives.u,
        u_x=derivatives.u_x,
        u_t=derivatives.u_t,
        u_xx=derivatives.u_xx,
        residual=residual,
    )


def cole_hopf_burgers_solution(
    x: np.ndarray,
    t: float,
    viscosity: float = 0.01 / math.pi,
    *,
    quadrature_order: int = 120,
) -> np.ndarray:
    """Evaluate the standard Burgers benchmark through Cole-Hopf.

    For

        u_t + u*u_x = viscosity*u_xx,
        u(x,0) = -sin(pi*x),

    the Cole-Hopf transformation gives a heat-kernel integral. After the
    substitution eta = 2*sqrt(viscosity*t)*z, both numerator and denominator
    become Gauss-Hermite integrals with weight exp(-z^2).

    This implementation uses a row-wise exponential shift to avoid overflow
    for the common benchmark viscosity viscosity = 0.01/pi.
    """
    _validate_viscosity(viscosity)

    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    if t < 0.0:
        raise ValueError("t must be non-negative.")

    if quadrature_order < 20:
        raise ValueError(
            "quadrature_order must be at least 20 for this benchmark."
        )

    if t == 0.0:
        return -np.sin(math.pi * x)

    nodes, weights = hermgauss(quadrature_order)

    shift = (
        2.0
        * math.sqrt(viscosity * t)
        * nodes
    )

    shifted_position = (
        x[:, None]
        - shift[None, :]
    )

    exponent = (
        -np.cos(
            math.pi * shifted_position
        )
        / (2.0 * math.pi * viscosity)
    )

    # Multiplying numerator and denominator by the same exp(-row_max)
    # stabilizes the ratio without changing the result.
    row_maximum = np.max(
        exponent,
        axis=1,
        keepdims=True,
    )

    stabilized_weight = (
        weights[None, :]
        * np.exp(
            exponent - row_maximum
        )
    )

    denominator = np.sum(
        stabilized_weight,
        axis=1,
    )

    numerator = np.sum(
        stabilized_weight
        * np.sin(
            math.pi * shifted_position
        ),
        axis=1,
    )

    return -numerator / denominator


def solve_burgers_reference(
    viscosity: float = 0.01 / math.pi,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    spatial_points: int = 401,
    evaluation_times: Sequence[float] | None = None,
    quadrature_order: int = 120,
) -> BurgersReferenceSolution:
    """Generate a dense Cole-Hopf reference grid."""
    _validate_viscosity(viscosity)

    if not x_min < x_max:
        raise ValueError("x_min must be smaller than x_max.")

    if spatial_points < 5:
        raise ValueError("spatial_points must be at least 5.")

    if spatial_points % 2 == 0:
        raise ValueError(
            "Use an odd number of spatial points so x=0 is included."
        )

    x = np.linspace(
        x_min,
        x_max,
        spatial_points,
        dtype=np.float64,
    )

    if evaluation_times is None:
        t = np.linspace(
            0.0,
            1.0,
            101,
            dtype=np.float64,
        )
    else:
        t = np.asarray(
            evaluation_times,
            dtype=np.float64,
        )

    if t.ndim != 1 or t.size < 2:
        raise ValueError(
            "evaluation_times must be one-dimensional with at least two values."
        )

    if np.any(np.diff(t) <= 0.0):
        raise ValueError(
            "evaluation_times must be strictly increasing."
        )

    if t[0] < 0.0:
        raise ValueError(
            "evaluation_times cannot contain negative times."
        )

    solution = np.empty(
        (
            x.size,
            t.size,
        ),
        dtype=np.float64,
    )

    for time_index, time_value in enumerate(t):
        solution[:, time_index] = (
            cole_hopf_burgers_solution(
                x=x,
                t=float(time_value),
                viscosity=viscosity,
                quadrature_order=quadrature_order,
            )
        )

    return BurgersReferenceSolution(
        x=x,
        t=t,
        u=solution,
        viscosity=float(viscosity),
        quadrature_order=quadrature_order,
        reference_method=(
            "Cole-Hopf heat-kernel integral "
            "with Gauss-Hermite quadrature"
        ),
    )


def quadrature_convergence_check(
    viscosity: float = 0.01 / math.pi,
    *,
    low_order: int = 40,
    medium_order: int = 80,
    high_order: int = 120,
    spatial_points: int = 201,
    evaluation_times: Sequence[float] | None = None,
) -> dict[str, float]:
    """Check convergence as the Gauss-Hermite order increases."""
    if not low_order < medium_order < high_order:
        raise ValueError(
            "Require low_order < medium_order < high_order."
        )

    if evaluation_times is None:
        evaluation_times = np.linspace(
            0.0,
            1.0,
            11,
        )

    low = solve_burgers_reference(
        viscosity=viscosity,
        spatial_points=spatial_points,
        evaluation_times=evaluation_times,
        quadrature_order=low_order,
    )
    medium = solve_burgers_reference(
        viscosity=viscosity,
        spatial_points=spatial_points,
        evaluation_times=evaluation_times,
        quadrature_order=medium_order,
    )
    high = solve_burgers_reference(
        viscosity=viscosity,
        spatial_points=spatial_points,
        evaluation_times=evaluation_times,
        quadrature_order=high_order,
    )

    low_to_medium = relative_l2_difference(
        low.u,
        medium.u,
    )
    medium_to_high = relative_l2_difference(
        medium.u,
        high.u,
    )

    return {
        "low_to_medium_relative_l2": low_to_medium,
        "medium_to_high_relative_l2": medium_to_high,
        "convergence_ratio": (
            low_to_medium / medium_to_high
            if medium_to_high > 0.0
            else math.inf
        ),
    }


def relative_l2_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return ||first-second||_2 / ||second||_2."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if first.shape != second.shape:
        raise ValueError("Compared arrays must have identical shapes.")

    denominator = np.linalg.norm(second)

    if denominator <= 0.0:
        raise ValueError("The reference array has zero norm.")

    return float(
        np.linalg.norm(first - second)
        / denominator
    )


def save_burgers_reference(
    reference: BurgersReferenceSolution,
    output_path: str | Path,
) -> None:
    """Save reference arrays and metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        x=reference.x,
        t=reference.t,
        u=reference.u,
        viscosity=np.array(
            reference.viscosity,
            dtype=np.float64,
        ),
        quadrature_order=np.array(
            reference.quadrature_order,
            dtype=np.int64,
        ),
        reference_method=np.array(
            reference.reference_method,
        ),
    )


def _validate_viscosity(viscosity: float) -> None:
    """Require a positive physical viscosity."""
    if not isinstance(viscosity, (int, float)):
        raise TypeError("viscosity must be a real number.")

    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive.")


class ManufacturedBurgersModel(nn.Module):
    """Known smooth function used to validate the residual code.

    We choose

        u(x,t) = exp(-t) sin(pi*x).

    It is not an unforced Burgers solution. Its analytically known residual
    is compared with the autograd residual.
    """

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        return (
            torch.exp(-t)
            * torch.sin(math.pi * x)
        )


def manufactured_burgers_residual(
    x: torch.Tensor,
    t: torch.Tensor,
    viscosity: float,
) -> torch.Tensor:
    """Return the analytic residual of the manufactured function."""
    u = (
        torch.exp(-t)
        * torch.sin(math.pi * x)
    )
    u_t = -u
    u_x = (
        math.pi
        * torch.exp(-t)
        * torch.cos(math.pi * x)
    )
    u_xx = -(math.pi**2) * u

    return (
        u_t
        + u * u_x
        - viscosity * u_xx
    )


if __name__ == "__main__":
    dtype = torch.float64
    viscosity = 0.01 / math.pi

    x_test = torch.tensor(
        [
            [-0.75],
            [-0.20],
            [0.30],
            [0.80],
        ],
        dtype=dtype,
    )
    t_test = torch.tensor(
        [
            [0.00],
            [0.15],
            [0.45],
            [0.90],
        ],
        dtype=dtype,
    )

    manufactured_model = ManufacturedBurgersModel()

    automatic = burgers_equation_residual(
        model=manufactured_model,
        x=x_test,
        t=t_test,
        viscosity=viscosity,
    )
    analytic = manufactured_burgers_residual(
        x=x_test,
        t=t_test,
        viscosity=viscosity,
    )

    assert torch.allclose(
        automatic.residual,
        analytic,
        atol=1.0e-11,
        rtol=1.0e-11,
    )

    convergence = quadrature_convergence_check(
        viscosity=viscosity,
        low_order=40,
        medium_order=80,
        high_order=120,
        spatial_points=201,
        evaluation_times=np.linspace(
            0.0,
            1.0,
            11,
        ),
    )

    assert (
        convergence[
            "medium_to_high_relative_l2"
        ]
        <
        convergence[
            "low_to_medium_relative_l2"
        ]
    )

    reference = solve_burgers_reference(
        viscosity=viscosity,
        spatial_points=401,
        evaluation_times=np.linspace(
            0.0,
            1.0,
            101,
        ),
        quadrature_order=120,
    )

    required_initial = -np.sin(
        math.pi * reference.x
    )

    assert np.allclose(
        reference.u[:, 0],
        required_initial,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    assert np.allclose(
        reference.u[0, :],
        0.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.allclose(
        reference.u[-1, :],
        0.0,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.all(
        np.isfinite(reference.u)
    )

    output_path = Path(
        "data/burgers_reference.npz"
    )

    save_burgers_reference(
        reference=reference,
        output_path=output_path,
    )

    print(
        "Burgers-equation scientific self-check passed."
    )
    print(
        "Autograd residual matches the manufactured analytic residual."
    )
    print(
        "Cole-Hopf initial and boundary conditions passed."
    )
    print(
        "40 -> 80 quadrature relative L2 change:  "
        f"{convergence['low_to_medium_relative_l2']:.6e}"
    )
    print(
        "80 -> 120 quadrature relative L2 change: "
        f"{convergence['medium_to_high_relative_l2']:.6e}"
    )
    print(
        "Quadrature convergence ratio: "
        f"{convergence['convergence_ratio']:.3e}"
    )
    print(
        "Saved 401 x 101 Burgers reference to: "
        f"{output_path}"
    )
