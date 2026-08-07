from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossWeights:
    """Weights used to balance the three PINN loss components."""

    pde: float = 1.0
    initial: float = 1.0
    boundary: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("pde", self.pde),
            ("initial", self.initial),
            ("boundary", self.boundary),
        ):
            if value < 0.0:
                raise ValueError(
                    f"{name} loss weight must be non-negative, "
                    f"but received {value}."
                )


@dataclass
class PINNLoss:
    """Differentiable loss values used to train a PINN."""

    total: torch.Tensor
    pde: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor


def mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the mean squared error between two tensors.

    The definition is

        MSE = (1 / N) * sum_i (prediction_i - target_i)^2.
    """
    _validate_same_shape(
        first=prediction,
        second=target,
        first_name="prediction",
        second_name="target",
    )

    return torch.mean((prediction - target) ** 2)


def pde_residual_loss(
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return the physics loss produced by the PDE residual.

    For the Heat Equation,

        residual = u_t - alpha*u_xx.

    An exact solution has residual = 0, so the loss is

        L_PDE = mean(residual^2).
    """
    _validate_column_tensor(
        tensor=residual,
        name="residual",
    )

    return torch.mean(residual**2)


def initial_condition_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the loss associated with the initial condition.

    For our benchmark,

        target = sin(pi*x)

    at t = 0.
    """
    return mean_squared_error(
        prediction=prediction,
        target=target,
    )


def boundary_condition_loss(
    left_prediction: torch.Tensor,
    left_target: torch.Tensor,
    right_prediction: torch.Tensor,
    right_target: torch.Tensor,
) -> torch.Tensor:
    """Return the combined loss for the two spatial boundaries.

    The two boundary errors are computed separately and then added:

        L_BC = MSE_left + MSE_right.

    For zero Dirichlet boundary conditions,

        left_target = 0,
        right_target = 0.
    """
    left_loss = mean_squared_error(
        prediction=left_prediction,
        target=left_target,
    )
    right_loss = mean_squared_error(
        prediction=right_prediction,
        target=right_target,
    )

    return left_loss + right_loss


def zero_dirichlet_boundary_loss(
    left_prediction: torch.Tensor,
    right_prediction: torch.Tensor,
) -> torch.Tensor:
    """Return the boundary loss for u(x_left,t)=u(x_right,t)=0."""
    return boundary_condition_loss(
        left_prediction=left_prediction,
        left_target=torch.zeros_like(left_prediction),
        right_prediction=right_prediction,
        right_target=torch.zeros_like(right_prediction),
    )


def combine_pinn_losses(
    pde: torch.Tensor,
    initial: torch.Tensor,
    boundary: torch.Tensor,
    weights: LossWeights | None = None,
) -> PINNLoss:
    """Combine the three differentiable PINN loss components.

    The total loss is

        L_total
        = lambda_PDE * L_PDE
        + lambda_IC  * L_IC
        + lambda_BC  * L_BC.

    No ``detach()`` or ``item()`` operation is used here because the
    computational graph must remain intact for backpropagation.
    """
    if weights is None:
        weights = LossWeights()

    _validate_scalar_loss(pde, "pde")
    _validate_scalar_loss(initial, "initial")
    _validate_scalar_loss(boundary, "boundary")

    total = (
        weights.pde * pde
        + weights.initial * initial
        + weights.boundary * boundary
    )

    return PINNLoss(
        total=total,
        pde=pde,
        initial=initial,
        boundary=boundary,
    )


def _validate_column_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Validate a tensor with shape (N, 1)."""
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape (N, 1), "
            f"but received {tuple(tensor.shape)}."
        )

    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")

    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating-point dtype.")


def _validate_same_shape(
    first: torch.Tensor,
    second: torch.Tensor,
    first_name: str,
    second_name: str,
) -> None:
    """Validate two compatible floating-point tensors."""
    _validate_column_tensor(first, first_name)
    _validate_column_tensor(second, second_name)

    if first.shape != second.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have the same shape, "
            f"but received {tuple(first.shape)} and {tuple(second.shape)}."
        )

    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must be on the same device."
        )

    if first.dtype != second.dtype:
        raise ValueError(
            f"{first_name} and {second_name} must have the same dtype."
        )


def _validate_scalar_loss(
    loss: torch.Tensor,
    name: str,
) -> None:
    """Validate a scalar differentiable loss tensor."""
    if loss.ndim != 0:
        raise ValueError(
            f"{name} loss must be a scalar tensor, "
            f"but received shape {tuple(loss.shape)}."
        )

    if not torch.is_floating_point(loss):
        raise TypeError(f"{name} loss must use a floating-point dtype.")


if __name__ == "__main__":
    # Scientific self-check with values whose losses can be computed by hand.
    dtype = torch.float64

    # PDE residuals: [1, -1]
    # L_PDE = (1^2 + (-1)^2) / 2 = 1
    residual = torch.tensor(
        [[1.0], [-1.0]],
        dtype=dtype,
        requires_grad=True,
    )
    pde = pde_residual_loss(residual)

    # Initial-condition predictions: [0, 1]
    # Targets: [0, 0]
    # L_IC = (0^2 + 1^2) / 2 = 0.5
    initial_prediction = torch.tensor(
        [[0.0], [1.0]],
        dtype=dtype,
        requires_grad=True,
    )
    initial_target = torch.zeros_like(initial_prediction)
    initial = initial_condition_loss(
        prediction=initial_prediction,
        target=initial_target,
    )

    # Boundary predictions are 0.5 and -0.5, with zero targets.
    # L_BC = 0.5^2 + (-0.5)^2 = 0.5
    left_prediction = torch.tensor(
        [[0.5]],
        dtype=dtype,
        requires_grad=True,
    )
    right_prediction = torch.tensor(
        [[-0.5]],
        dtype=dtype,
        requires_grad=True,
    )
    boundary = zero_dirichlet_boundary_loss(
        left_prediction=left_prediction,
        right_prediction=right_prediction,
    )

    losses = combine_pinn_losses(
        pde=pde,
        initial=initial,
        boundary=boundary,
        weights=LossWeights(
            pde=1.0,
            initial=1.0,
            boundary=1.0,
        ),
    )

    expected_pde = torch.tensor(1.0, dtype=dtype)
    expected_initial = torch.tensor(0.5, dtype=dtype)
    expected_boundary = torch.tensor(0.5, dtype=dtype)
    expected_total = torch.tensor(2.0, dtype=dtype)

    assert torch.allclose(losses.pde, expected_pde)
    assert torch.allclose(losses.initial, expected_initial)
    assert torch.allclose(losses.boundary, expected_boundary)
    assert torch.allclose(losses.total, expected_total)

    # Confirm that the combined loss remains differentiable.
    losses.total.backward()

    assert residual.grad is not None
    assert initial_prediction.grad is not None
    assert left_prediction.grad is not None
    assert right_prediction.grad is not None

    print("PINN loss scientific self-check passed.")
    print(f"PDE loss:      {losses.pde.item():.6f}")
    print(f"Initial loss:  {losses.initial.item():.6f}")
    print(f"Boundary loss: {losses.boundary.item():.6f}")
    print(f"Total loss:    {losses.total.item():.6f}")
