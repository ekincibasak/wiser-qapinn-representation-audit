from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PINNDerivatives:
    """Container for a PINN prediction and its physical derivatives."""

    u: torch.Tensor
    u_x: torch.Tensor
    u_t: torch.Tensor
    u_xx: torch.Tensor


def compute_pinn_derivatives(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> PINNDerivatives:
    """Calculate u, u_x, u_t, and u_xx for a PINN model.

    The model is assumed to represent

        u_theta(x, t).

    Parameters
    ----------
    model:
        A PyTorch model that receives a tensor with two columns [x, t]
        and returns one scalar prediction per point.
    x:
        Spatial coordinates with shape (N, 1).
    t:
        Time coordinates with shape (N, 1).

    Returns
    -------
    PINNDerivatives
        A dataclass containing:
        - u: predicted solution u_theta(x, t)
        - u_x: first spatial derivative
        - u_t: first time derivative
        - u_xx: second spatial derivative

    Notes
    -----
    ``create_graph=True`` is essential because:
    1. We need to differentiate u_x again to obtain u_xx.
    2. The PDE loss built from these derivatives must later be
       differentiated with respect to the model parameters.
    """
    _validate_coordinate_tensors(x=x, t=t)

    # Combine the physical coordinates into one model input:
    #
    #     xt = [x, t]
    #
    # We clone and detach the input so that this function creates a clean
    # leaf tensor whose derivatives can be tracked reliably.
    xt = torch.cat([x, t], dim=1).clone().detach().requires_grad_(True)

    # Neural-network prediction:
    #
    #     u = u_theta(x, t)
    u = model(xt)

    if u.ndim != 2 or u.shape != (xt.shape[0], 1):
        raise ValueError(
            "The model output must have shape (N, 1), "
            f"but received {tuple(u.shape)}."
        )

    # First derivatives:
    #
    #     grad_u = [du/dx, du/dt]
    #
    # For a feed-forward model, each row is evaluated independently.
    # Therefore, differentiating the sum of the outputs gives the
    # pointwise derivatives for each input row.
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # Second spatial derivative:
    #
    #     u_xx = d/dx (u_x)
    grad_u_x = torch.autograd.grad(
        outputs=u_x,
        inputs=xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = grad_u_x[:, 0:1]

    return PINNDerivatives(
        u=u,
        u_x=u_x,
        u_t=u_t,
        u_xx=u_xx,
    )


def _validate_coordinate_tensors(
    x: torch.Tensor,
    t: torch.Tensor,
) -> None:
    """Validate the coordinate tensors used by the PINN."""
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(
            "x must have shape (N, 1), "
            f"but received {tuple(x.shape)}."
        )

    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(
            "t must have shape (N, 1), "
            f"but received {tuple(t.shape)}."
        )

    if x.shape != t.shape:
        raise ValueError(
            "x and t must have the same shape, "
            f"but received {tuple(x.shape)} and {tuple(t.shape)}."
        )

    if x.device != t.device:
        raise ValueError("x and t must be on the same device.")

    if x.dtype != t.dtype:
        raise ValueError("x and t must have the same dtype.")

    if not torch.is_floating_point(x) or not torch.is_floating_point(t):
        raise TypeError("x and t must use a floating-point dtype.")


class AnalyticTestFunction(nn.Module):
    """Known function used only to verify automatic differentiation.

    The function is

        u(x, t) = x^2 + 3t.

    Therefore,

        u_x  = 2x,
        u_t  = 3,
        u_xx = 2.
    """

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return x**2 + 3.0 * t


if __name__ == "__main__":
    # This self-check verifies the derivative code using a function whose
    # derivatives are known exactly.
    test_model = AnalyticTestFunction()

    x_values = torch.tensor(
        [[0.0], [0.5], [1.0]],
        dtype=torch.float64,
    )
    t_values = torch.tensor(
        [[0.0], [0.2], [0.8]],
        dtype=torch.float64,
    )

    derivatives = compute_pinn_derivatives(
        model=test_model,
        x=x_values,
        t=t_values,
    )

    expected_u_x = 2.0 * x_values
    expected_u_t = torch.full_like(t_values, 3.0)
    expected_u_xx = torch.full_like(x_values, 2.0)

    assert torch.allclose(
        derivatives.u_x,
        expected_u_x,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        derivatives.u_t,
        expected_u_t,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        derivatives.u_xx,
        expected_u_xx,
        atol=1e-10,
        rtol=1e-10,
    )

    print("Automatic-differentiation self-check passed.")
    print("u:")
    print(derivatives.u)
    print("u_x:")
    print(derivatives.u_x)
    print("u_t:")
    print(derivatives.u_t)
    print("u_xx:")
    print(derivatives.u_xx)
