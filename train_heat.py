from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    # Imports used when the files are installed as a package.
    from wiser_qapinn.models.classical_mlp import ClassicalMLP
    from wiser_qapinn.physics.heat import (
        heat_equation_residual,
        heat_exact_solution,
        heat_initial_condition,
    )
    from wiser_qapinn.training.losses import (
        LossWeights,
        combine_pinn_losses,
        initial_condition_loss,
        pde_residual_loss,
        zero_dirichlet_boundary_loss,
    )
except ImportError:
    # Fallback imports used when the five files are in the same folder.
    from classical_mlp import ClassicalMLP
    from heat import (
        heat_equation_residual,
        heat_exact_solution,
        heat_initial_condition,
    )
    from losses import (
        LossWeights,
        combine_pinn_losses,
        initial_condition_loss,
        pde_residual_loss,
        zero_dirichlet_boundary_loss,
    )


@dataclass(frozen=True)
class HeatTrainingConfig:
    """Configuration for reproducible Heat Equation PINN training."""

    alpha: float = 0.1
    epochs: int = 1000
    learning_rate: float = 1.0e-3

    n_collocation: int = 1000
    n_initial: int = 128
    n_boundary: int = 128

    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    pde_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0

    seed: int = 42
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if min(
            self.n_collocation,
            self.n_initial,
            self.n_boundary,
        ) <= 0:
            raise ValueError("All sample counts must be positive.")
        if not self.x_min < self.x_max:
            raise ValueError("x_min must be smaller than x_max.")
        if not self.t_min < self.t_max:
            raise ValueError("t_min must be smaller than t_max.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


@dataclass
class HeatTrainingPoints:
    """Fixed space-time samples used during one training run."""

    x_f: torch.Tensor
    t_f: torch.Tensor

    x_i: torch.Tensor
    t_i: torch.Tensor

    x_left: torch.Tensor
    t_left: torch.Tensor

    x_right: torch.Tensor
    t_right: torch.Tensor


@dataclass
class TrainingHistory:
    """Loss and gradient values recorded during optimization."""

    epoch: list[int]
    total_loss: list[float]
    pde_loss: list[float]
    initial_loss: list[float]
    boundary_loss: list[float]
    gradient_norm: list[float]

    @classmethod
    def empty(cls) -> "TrainingHistory":
        return cls(
            epoch=[],
            total_loss=[],
            pde_loss=[],
            initial_loss=[],
            boundary_loss=[],
            gradient_norm=[],
        )

    def append(
        self,
        epoch: int,
        total: torch.Tensor,
        pde: torch.Tensor,
        initial: torch.Tensor,
        boundary: torch.Tensor,
        gradient_norm: float,
    ) -> None:
        self.epoch.append(epoch)
        self.total_loss.append(float(total.detach().cpu()))
        self.pde_loss.append(float(pde.detach().cpu()))
        self.initial_loss.append(float(initial.detach().cpu()))
        self.boundary_loss.append(float(boundary.detach().cpu()))
        self.gradient_norm.append(float(gradient_norm))

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Simple accuracy metrics against the analytical Heat solution."""

    relative_l2: float
    mean_absolute_error: float
    maximum_absolute_error: float


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_heat_training_points(
    config: HeatTrainingConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> HeatTrainingPoints:
    """Sample fixed collocation, initial, and boundary points.

    The points are sampled once and then kept fixed during optimization.
    This makes one experiment deterministic after fixing the random seed.
    """
    x_f = uniform_column(
        n=config.n_collocation,
        lower=config.x_min,
        upper=config.x_max,
        device=device,
        dtype=dtype,
    )
    t_f = uniform_column(
        n=config.n_collocation,
        lower=config.t_min,
        upper=config.t_max,
        device=device,
        dtype=dtype,
    )

    x_i = uniform_column(
        n=config.n_initial,
        lower=config.x_min,
        upper=config.x_max,
        device=device,
        dtype=dtype,
    )
    t_i = torch.full_like(x_i, config.t_min)

    t_boundary = uniform_column(
        n=config.n_boundary,
        lower=config.t_min,
        upper=config.t_max,
        device=device,
        dtype=dtype,
    )
    x_left = torch.full_like(t_boundary, config.x_min)
    x_right = torch.full_like(t_boundary, config.x_max)

    return HeatTrainingPoints(
        x_f=x_f,
        t_f=t_f,
        x_i=x_i,
        t_i=t_i,
        x_left=x_left,
        t_left=t_boundary,
        x_right=x_right,
        t_right=t_boundary.clone(),
    )


def uniform_column(
    n: int,
    lower: float,
    upper: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample an (n, 1) tensor uniformly from [lower, upper]."""
    return (
        lower
        + (upper - lower)
        * torch.rand(
            (n, 1),
            device=device,
            dtype=dtype,
        )
    )


def calculate_heat_losses(
    model: nn.Module,
    points: HeatTrainingPoints,
    config: HeatTrainingConfig,
):
    """Calculate all differentiable Heat PINN loss components."""
    equation = heat_equation_residual(
        model=model,
        x=points.x_f,
        t=points.t_f,
        alpha=config.alpha,
    )
    pde = pde_residual_loss(equation.residual)

    initial_inputs = torch.cat(
        [points.x_i, points.t_i],
        dim=1,
    )
    initial_prediction = model(initial_inputs)
    initial_target = heat_initial_condition(points.x_i)

    initial = initial_condition_loss(
        prediction=initial_prediction,
        target=initial_target,
    )

    left_inputs = torch.cat(
        [points.x_left, points.t_left],
        dim=1,
    )
    right_inputs = torch.cat(
        [points.x_right, points.t_right],
        dim=1,
    )

    left_prediction = model(left_inputs)
    right_prediction = model(right_inputs)

    boundary = zero_dirichlet_boundary_loss(
        left_prediction=left_prediction,
        right_prediction=right_prediction,
    )

    return combine_pinn_losses(
        pde=pde,
        initial=initial,
        boundary=boundary,
        weights=LossWeights(
            pde=config.pde_weight,
            initial=config.initial_weight,
            boundary=config.boundary_weight,
        ),
    )


def global_gradient_norm(model: nn.Module) -> float:
    """Return the Euclidean norm of all parameter gradients."""
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += float(
                torch.sum(parameter.grad.detach() ** 2).cpu()
            )

    return math.sqrt(squared_norm)


def train_heat_pinn(
    model: nn.Module,
    config: HeatTrainingConfig,
    device: torch.device | str = "cpu",
) -> TrainingHistory:
    """Train a PINN for the one-dimensional Heat Equation."""
    device = torch.device(device)
    model.to(device)

    set_reproducible_seed(config.seed)

    points = sample_heat_training_points(
        config=config,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = TrainingHistory.empty()

    for epoch in range(1, config.epochs + 1):
        # Remove gradients left over from the preceding epoch.
        optimizer.zero_grad(set_to_none=True)

        # Forward calculation of the PDE, IC, and BC losses.
        losses = calculate_heat_losses(
            model=model,
            points=points,
            config=config,
        )

        # Backpropagation: calculate dL_total/dtheta.
        losses.total.backward()

        gradient_norm = global_gradient_norm(model)

        # Update all trainable weights and biases.
        optimizer.step()

        history.append(
            epoch=epoch,
            total=losses.total,
            pde=losses.pde,
            initial=losses.initial,
            boundary=losses.boundary,
            gradient_norm=gradient_norm,
        )

        if (
            epoch == 1
            or epoch % config.log_every == 0
            or epoch == config.epochs
        ):
            print(
                f"Epoch {epoch:5d}/{config.epochs} | "
                f"total={history.total_loss[-1]:.6e} | "
                f"pde={history.pde_loss[-1]:.6e} | "
                f"ic={history.initial_loss[-1]:.6e} | "
                f"bc={history.boundary_loss[-1]:.6e} | "
                f"grad={history.gradient_norm[-1]:.6e}"
            )

    return history


@torch.no_grad()
def evaluate_heat_pinn(
    model: nn.Module,
    alpha: float,
    device: torch.device | str = "cpu",
    n_x: int = 101,
    n_t: int = 51,
) -> EvaluationMetrics:
    """Compare the trained model with the analytical Heat solution."""
    if n_x < 2 or n_t < 2:
        raise ValueError("n_x and n_t must both be at least 2.")

    device = torch.device(device)
    dtype = next(model.parameters()).dtype

    x = torch.linspace(
        0.0,
        1.0,
        n_x,
        device=device,
        dtype=dtype,
    )
    t = torch.linspace(
        0.0,
        1.0,
        n_t,
        device=device,
        dtype=dtype,
    )

    x_grid, t_grid = torch.meshgrid(
        x,
        t,
        indexing="ij",
    )

    x_column = x_grid.reshape(-1, 1)
    t_column = t_grid.reshape(-1, 1)

    model_inputs = torch.cat(
        [x_column, t_column],
        dim=1,
    )

    prediction = model(model_inputs)
    reference = heat_exact_solution(
        x=x_column,
        t=t_column,
        alpha=alpha,
    )

    error = prediction - reference

    relative_l2 = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(reference)
    )
    mae = torch.mean(torch.abs(error))
    max_error = torch.max(torch.abs(error))

    return EvaluationMetrics(
        relative_l2=float(relative_l2.cpu()),
        mean_absolute_error=float(mae.cpu()),
        maximum_absolute_error=float(max_error.cpu()),
    )


def parameter_change_norm(
    initial_parameters: list[torch.Tensor],
    model: nn.Module,
) -> float:
    """Measure how much the optimizer changed the model parameters."""
    squared_change = 0.0

    for initial, current in zip(
        initial_parameters,
        model.parameters(),
        strict=True,
    ):
        difference = current.detach().cpu() - initial
        squared_change += float(torch.sum(difference**2))

    return math.sqrt(squared_change)


def save_training_results(
    output_directory: str | Path,
    config: HeatTrainingConfig,
    history: TrainingHistory,
    metrics: EvaluationMetrics,
    model: nn.Module,
) -> None:
    """Save configuration, history, metrics, and trained parameters."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(config), file, indent=2)

    with (output_directory / "history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history.to_dict(), file, indent=2)

    with (output_directory / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(metrics), file, indent=2)

    torch.save(
        model.state_dict(),
        output_directory / "model_state.pt",
    )


if __name__ == "__main__":
    # This is a small, reproducible demonstration.
    # It proves that training occurs; it is not yet our final WISER experiment.
    config = HeatTrainingConfig(
        epochs=500,
        learning_rate=1.0e-3,
        n_collocation=512,
        n_initial=128,
        n_boundary=128,
        seed=42,
        log_every=100,
    )

    set_reproducible_seed(config.seed)

    model = ClassicalMLP(
        in_features=2,
        out_features=1,
        hidden_layers=(16, 16),
    )

    initial_parameters = [
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    ]

    history = train_heat_pinn(
        model=model,
        config=config,
        device="cpu",
    )

    metrics = evaluate_heat_pinn(
        model=model,
        alpha=config.alpha,
        device="cpu",
    )

    change_norm = parameter_change_norm(
        initial_parameters=initial_parameters,
        model=model,
    )

    assert history.total_loss[-1] < history.total_loss[0], (
        "The demonstration training did not reduce the total loss."
    )
    assert change_norm > 0.0, (
        "The model parameters did not change during training."
    )
    assert all(
        math.isfinite(value)
        for value in history.total_loss
    ), "A non-finite loss was encountered."

    output_directory = Path("outputs/heat_training_demo")
    save_training_results(
        output_directory=output_directory,
        config=config,
        history=history,
        metrics=metrics,
        model=model,
    )

    print("\nTraining self-check passed.")
    print(
        "Initial total loss: "
        f"{history.total_loss[0]:.6e}"
    )
    print(
        "Final total loss:   "
        f"{history.total_loss[-1]:.6e}"
    )
    print(
        "Parameter-change norm: "
        f"{change_norm:.6e}"
    )
    print(
        "Relative L2 error: "
        f"{metrics.relative_l2:.6e}"
    )
    print(
        "Mean absolute error: "
        f"{metrics.mean_absolute_error:.6e}"
    )
    print(
        "Maximum absolute error: "
        f"{metrics.maximum_absolute_error:.6e}"
    )
    print(
        "Saved demonstration outputs to: "
        f"{output_directory}"
    )
