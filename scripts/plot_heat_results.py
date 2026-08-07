from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from wiser_qapinn.models.classical_mlp import ClassicalMLP
    from wiser_qapinn.physics.heat import heat_exact_solution
except ImportError:
    from classical_mlp import ClassicalMLP
    from heat import heat_exact_solution


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch state dictionary safely when supported."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def reconstruct_model(
    model_state_path: Path,
    hidden_layers: Sequence[int],
) -> ClassicalMLP:
    """Reconstruct the trained MLP and load its learned parameters."""
    model = ClassicalMLP(
        in_features=2,
        out_features=1,
        hidden_layers=tuple(hidden_layers),
    )

    state_dict = load_model_state(model_state_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def evaluate_on_grid(
    model: torch.nn.Module,
    alpha: float,
    n_x: int,
    n_t: int,
) -> dict[str, np.ndarray | float]:
    """Evaluate the trained PINN and analytical solution on a grid."""
    if n_x < 2 or n_t < 2:
        raise ValueError("n_x and n_t must both be at least 2.")

    dtype = next(model.parameters()).dtype

    x = torch.linspace(
        0.0,
        1.0,
        n_x,
        dtype=dtype,
    )
    t = torch.linspace(
        0.0,
        1.0,
        n_t,
        dtype=dtype,
    )

    x_grid, t_grid = torch.meshgrid(
        x,
        t,
        indexing="ij",
    )

    x_column = x_grid.reshape(-1, 1)
    t_column = t_grid.reshape(-1, 1)

    prediction = model(
        torch.cat([x_column, t_column], dim=1)
    )
    reference = heat_exact_solution(
        x=x_column,
        t=t_column,
        alpha=alpha,
    )

    prediction_grid = prediction.reshape(n_x, n_t)
    reference_grid = reference.reshape(n_x, n_t)
    absolute_error_grid = torch.abs(
        prediction_grid - reference_grid
    )

    relative_l2 = (
        torch.linalg.vector_norm(
            prediction_grid - reference_grid
        )
        / torch.linalg.vector_norm(reference_grid)
    )
    mean_absolute_error = torch.mean(absolute_error_grid)
    maximum_absolute_error = torch.max(absolute_error_grid)

    return {
        "x": x.cpu().numpy(),
        "t": t.cpu().numpy(),
        "prediction": prediction_grid.cpu().numpy(),
        "reference": reference_grid.cpu().numpy(),
        "absolute_error": absolute_error_grid.cpu().numpy(),
        "relative_l2": float(relative_l2.cpu()),
        "mean_absolute_error": float(
            mean_absolute_error.cpu()
        ),
        "maximum_absolute_error": float(
            maximum_absolute_error.cpu()
        ),
    }


def save_loss_curves(
    history: dict,
    output_path: Path,
) -> None:
    """Save total and component PINN losses."""
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    axis.semilogy(
        history["epoch"],
        history["total_loss"],
        label="Total loss",
    )
    axis.semilogy(
        history["epoch"],
        history["pde_loss"],
        label="PDE loss",
    )
    axis.semilogy(
        history["epoch"],
        history["initial_loss"],
        label="Initial-condition loss",
    )
    axis.semilogy(
        history["epoch"],
        history["boundary_loss"],
        label="Boundary-condition loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Heat PINN training losses")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_gradient_curve(
    history: dict,
    output_path: Path,
) -> None:
    """Save the global gradient-norm history."""
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    axis.semilogy(
        history["epoch"],
        history["gradient_norm"],
        label="Global gradient norm",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Gradient norm")
    axis.set_title("Gradient flow during Heat PINN training")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_field_plot(
    x: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    """Save one space-time field as an independent figure."""
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    image = axis.pcolormesh(
        t,
        x,
        values,
        shading="auto",
    )

    axis.set_xlabel("Time t")
    axis.set_ylabel("Position x")
    axis.set_title(title)

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_solution_slices(
    x: np.ndarray,
    t: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    output_path: Path,
) -> None:
    """Compare predicted and exact spatial profiles at selected times."""
    selected_times = (0.0, 0.5, 1.0)

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    for requested_time in selected_times:
        time_index = int(
            np.argmin(np.abs(t - requested_time))
        )
        actual_time = float(t[time_index])

        axis.plot(
            x,
            reference[:, time_index],
            label=f"Exact, t={actual_time:.2f}",
        )
        axis.plot(
            x,
            prediction[:, time_index],
            linestyle="--",
            label=f"PINN, t={actual_time:.2f}",
        )

    axis.set_xlabel("Position x")
    axis.set_ylabel("Temperature u(x,t)")
    axis.set_title("Exact and PINN solution profiles")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_evaluation_data(
    evaluation: dict[str, np.ndarray | float],
    output_directory: Path,
) -> None:
    """Save grid arrays and independently recomputed metrics."""
    np.savez_compressed(
        output_directory / "evaluation_grid.npz",
        x=evaluation["x"],
        t=evaluation["t"],
        prediction=evaluation["prediction"],
        reference=evaluation["reference"],
        absolute_error=evaluation["absolute_error"],
    )

    metrics = {
        "relative_l2": evaluation["relative_l2"],
        "mean_absolute_error": (
            evaluation["mean_absolute_error"]
        ),
        "maximum_absolute_error": (
            evaluation["maximum_absolute_error"]
        ),
    }

    with (
        output_directory / "visualization_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def create_all_figures(
    results_directory: Path,
    output_directory: Path,
    hidden_layers: Sequence[int],
    n_x: int,
    n_t: int,
) -> dict[str, float]:
    """Create all scientific figures from a completed training run."""
    config_path = results_directory / "config.json"
    history_path = results_directory / "history.json"
    model_path = results_directory / "model_state.pt"

    for required_path in (
        config_path,
        history_path,
        model_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    config = load_json(config_path)
    history = load_json(history_path)

    model = reconstruct_model(
        model_state_path=model_path,
        hidden_layers=hidden_layers,
    )

    evaluation = evaluate_on_grid(
        model=model,
        alpha=float(config["alpha"]),
        n_x=n_x,
        n_t=n_t,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_loss_curves(
        history=history,
        output_path=output_directory / "loss_curves.png",
    )
    save_gradient_curve(
        history=history,
        output_path=output_directory / "gradient_norm.png",
    )
    save_field_plot(
        x=evaluation["x"],
        t=evaluation["t"],
        values=evaluation["prediction"],
        title="PINN-predicted Heat Equation solution",
        colorbar_label="Predicted temperature",
        output_path=output_directory / "predicted_solution.png",
    )
    save_field_plot(
        x=evaluation["x"],
        t=evaluation["t"],
        values=evaluation["reference"],
        title="Analytical Heat Equation solution",
        colorbar_label="Exact temperature",
        output_path=output_directory / "exact_solution.png",
    )
    save_field_plot(
        x=evaluation["x"],
        t=evaluation["t"],
        values=evaluation["absolute_error"],
        title="Absolute error of the Heat PINN",
        colorbar_label="Absolute error",
        output_path=output_directory / "absolute_error.png",
    )
    save_solution_slices(
        x=evaluation["x"],
        t=evaluation["t"],
        prediction=evaluation["prediction"],
        reference=evaluation["reference"],
        output_path=output_directory / "solution_slices.png",
    )

    save_evaluation_data(
        evaluation=evaluation,
        output_directory=output_directory,
    )

    return {
        "relative_l2": float(
            evaluation["relative_l2"]
        ),
        "mean_absolute_error": float(
            evaluation["mean_absolute_error"]
        ),
        "maximum_absolute_error": float(
            evaluation["maximum_absolute_error"]
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create scientific figures for a trained "
            "Heat Equation PINN."
        )
    )

    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("outputs/heat_training_demo"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/heat_figures"),
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[16, 16],
    )
    parser.add_argument(
        "--n-x",
        type=int,
        default=201,
    )
    parser.add_argument(
        "--n-t",
        type=int,
        default=101,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    metrics = create_all_figures(
        results_directory=arguments.results_directory,
        output_directory=arguments.output_directory,
        hidden_layers=arguments.hidden_layers,
        n_x=arguments.n_x,
        n_t=arguments.n_t,
    )

    print("Heat PINN figure generation passed.")
    print(
        "Relative L2 error: "
        f"{metrics['relative_l2']:.6e}"
    )
    print(
        "Mean absolute error: "
        f"{metrics['mean_absolute_error']:.6e}"
    )
    print(
        "Maximum absolute error: "
        f"{metrics['maximum_absolute_error']:.6e}"
    )
    print(
        "Figures saved to: "
        f"{arguments.output_directory}"
    )
