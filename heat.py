from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

try:
    # Import used when this file is inside the Python package.
    from .derivatives import PINNDerivatives, compute_pinn_derivatives
except ImportError:
    # Fallback used when heat.py and derivatives.py are run directly
    # from the same folder.
    from derivatives import PINNDerivatives, compute_pinn_derivatives


@dataclass
class HeatEquationResult:
    """Prediction, derivatives, and residual for the Heat Equation."""

    u: torch.Tensor
    u_x: torch.Tensor
    u_t: torch.Tensor
    u_xx: torch.Tensor
    residual: torch.Tensor


def heat_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Return the initial temperature profile u(x, 0) = sin(pi x)."""
    return torch.sin(math.pi * x)


def heat_exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return the analytical solution of the benchmark Heat Equation.

    The problem is

        u_t = alpha u_xx,

    on

        0 <= x <= 1,
        t >= 0,

    with

        u(x, 0) = sin(pi x),
        u(0, t) = 0,
        u(1, t) = 0.

    Its analytical solution is

        u(x, t) = exp(-alpha pi^2 t) sin(pi x).
    """
    _validate_alpha(alpha)
    _validate_matching_shapes(x=x, t=t)

    return (
        torch.exp(-alpha * math.pi**2 * t)
        * torch.sin(math.pi * x)
    )


def heat_equation_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
) -> HeatEquationResult:
    """Calculate the Heat Equation residual for a PINN prediction.

    The model represents u_theta(x, t).

    Moving all terms of the Heat Equation to the left gives

        r_theta(x, t)
        = u_t(x, t) - alpha u_xx(x, t).

    An exact solution satisfies r_theta(x, t) = 0.
    """
    _validate_alpha(alpha)

    derivatives: PINNDerivatives = compute_pinn_derivatives(
        model=model,
        x=x,
        t=t,
    )

    residual = (
        derivatives.u_t
        - alpha * derivatives.u_xx
    )

    return HeatEquationResult(
        u=derivatives.u,
        u_x=derivatives.u_x,
        u_t=derivatives.u_t,
        u_xx=derivatives.u_xx,
        residual=residual,
    )


def _validate_alpha(alpha: float) -> None:
    """Validate the thermal diffusivity."""
    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number.")

    if alpha <= 0.0:
        raise ValueError(
            "alpha must be positive for the standard Heat Equation."
        )


def _validate_matching_shapes(
    x: torch.Tensor,
    t: torch.Tensor,
) -> None:
    """Check that x and t have compatible shapes."""
    if x.shape != t.shape:
        raise ValueError(
            "x and t must have the same shape, "
            f"but received {tuple(x.shape)} and {tuple(t.shape)}."
        )


class ExactHeatSolutionModel(nn.Module):
    """PyTorch module representing the analytical Heat solution.

    This module is used only to validate the residual implementation.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        _validate_alpha(alpha)
        self.alpha = float(alpha)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        return (
            torch.exp(
                -self.alpha * math.pi**2 * t
            )
            * torch.sin(math.pi * x)
        )


if __name__ == "__main__":
    # Float64 improves the precision of this analytical validation.
    dtype = torch.float64
    alpha = 0.1

    # Space-time points used to check the PDE residual.
    x_values = torch.tensor(
        [[0.0], [0.25], [0.50], [0.75], [1.0]],
        dtype=dtype,
    )
    t_values = torch.tensor(
        [[0.0], [0.10], [0.20], [0.50], [1.0]],
        dtype=dtype,
    )

    exact_model = ExactHeatSolutionModel(alpha=alpha)

    result = heat_equation_residual(
        model=exact_model,
        x=x_values,
        t=t_values,
        alpha=alpha,
    )

    # The analytical solution must satisfy u_t - alpha*u_xx = 0.
    max_residual = torch.max(
        torch.abs(result.residual)
    ).item()

    assert max_residual < 1e-10, (
        "Heat residual validation failed. "
        f"Maximum absolute residual: {max_residual:.3e}"
    )

    # Validate the initial condition.
    x_initial = torch.linspace(
        0.0,
        1.0,
        101,
        dtype=dtype,
    ).reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)

    exact_initial = heat_exact_solution(
        x=x_initial,
        t=t_initial,
        alpha=alpha,
    )
    required_initial = heat_initial_condition(x_initial)

    assert torch.allclose(
        exact_initial,
        required_initial,
        atol=1e-12,
        rtol=1e-12,
    )

    # Validate the two zero Dirichlet boundary conditions.
    t_boundary = torch.linspace(
        0.0,
        1.0,
        101,
        dtype=dtype,
    ).reshape(-1, 1)

    x_left = torch.zeros_like(t_boundary)
    x_right = torch.ones_like(t_boundary)

    left_boundary = heat_exact_solution(
        x=x_left,
        t=t_boundary,
        alpha=alpha,
    )
    right_boundary = heat_exact_solution(
        x=x_right,
        t=t_boundary,
        alpha=alpha,
    )

    assert torch.allclose(
        left_boundary,
        torch.zeros_like(left_boundary),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        right_boundary,
        torch.zeros_like(right_boundary),
        atol=1e-12,
        rtol=1e-12,
    )

    print("Heat Equation scientific self-check passed.")
    print(f"Maximum |u_t - alpha*u_xx|: {max_residual:.3e}")
    print("Residual values:")
    print(result.residual)
