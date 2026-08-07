from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from burgers import (
    BurgersReferenceSolution,
    burgers_equation_residual,
    burgers_initial_condition,
)
from classical_bottleneck import ClassicalBottleneckPINN
from losses import (
    LossWeights,
    combine_pinn_losses,
    initial_condition_loss,
    pde_residual_loss,
    zero_dirichlet_boundary_loss,
)
from quantum_layer import HybridQuantumPINN
from spectra import FrequencyBands, analyze_spatial_profile
from train_heat import global_gradient_norm, set_reproducible_seed


@dataclass(frozen=True)
class BurgersTrainingConfig:
    """Configuration for matched classical-versus-quantum training."""

    viscosity: float = 0.01 / math.pi
    epochs: int = 100
    learning_rate: float = 1.0e-3

    n_collocation: int = 128
    n_initial: int = 32
    n_boundary: int = 32

    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    pde_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0

    log_every: int = 100

    def __post_init__(self) -> None:
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be positive.")
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
class BurgersTrainingPoints:
    """Fixed samples shared by all models within one random seed."""

    x_f: torch.Tensor
    t_f: torch.Tensor

    x_i: torch.Tensor
    t_i: torch.Tensor

    x_left: torch.Tensor
    t_left: torch.Tensor

    x_right: torch.Tensor
    t_right: torch.Tensor


@dataclass(frozen=True)
class ModelSpecification:
    """Definition of one model in the controlled pilot."""

    name: str
    architecture: str
    constructor: Callable[[], nn.Module]


@dataclass
class BurgersRunResult:
    """Scalar and spectral results from one model and one seed."""

    model: str
    architecture: str
    seed: int

    trainable_parameters: int
    quantum_parameters: int

    initial_total_loss: float
    final_total_loss: float
    final_pde_loss: float
    final_initial_loss: float
    final_boundary_loss: float

    final_global_gradient_norm: float
    final_quantum_gradient_norm: float | None
    quantum_parameter_change_norm: float | None

    relative_l2: float
    mean_absolute_error: float
    maximum_absolute_error: float

    training_seconds: float
    spectral_metrics: dict[str, Any]


def model_specifications() -> list[ModelSpecification]:
    """Return the classical bottleneck and four-qubit QAPINN."""
    return [
        ModelSpecification(
            name="classical_bottleneck",
            architecture="2 -> 8 -> 4 -> 8 -> 1",
            constructor=lambda: ClassicalBottleneckPINN(
                in_features=2,
                classical_width=8,
                bottleneck_dimension=4,
                out_features=1,
            ),
        ),
        ModelSpecification(
            name="four_qubit_qapinn",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "4-qubit VQC(depth=2, ring) -> "
                "4 <Z> -> 8 -> 1"
            ),
            constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
    ]


def uniform_column(
    n: int,
    lower: float,
    upper: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample an (n,1) tensor uniformly on [lower, upper]."""
    return (
        lower
        + (upper - lower)
        * torch.rand(
            (n, 1),
            dtype=dtype,
        )
    )


def sample_burgers_training_points(
    config: BurgersTrainingConfig,
    dtype: torch.dtype = torch.float32,
) -> BurgersTrainingPoints:
    """Sample one shared set of PDE, IC, and BC points."""
    x_f = uniform_column(
        config.n_collocation,
        config.x_min,
        config.x_max,
        dtype,
    )
    t_f = uniform_column(
        config.n_collocation,
        config.t_min,
        config.t_max,
        dtype,
    )

    x_i = uniform_column(
        config.n_initial,
        config.x_min,
        config.x_max,
        dtype,
    )
    t_i = torch.full_like(
        x_i,
        config.t_min,
    )

    t_boundary = uniform_column(
        config.n_boundary,
        config.t_min,
        config.t_max,
        dtype,
    )

    x_left = torch.full_like(
        t_boundary,
        config.x_min,
    )
    x_right = torch.full_like(
        t_boundary,
        config.x_max,
    )

    return BurgersTrainingPoints(
        x_f=x_f,
        t_f=t_f,
        x_i=x_i,
        t_i=t_i,
        x_left=x_left,
        t_left=t_boundary,
        x_right=x_right,
        t_right=t_boundary.clone(),
    )


def calculate_burgers_losses(
    model: nn.Module,
    points: BurgersTrainingPoints,
    config: BurgersTrainingConfig,
):
    """Calculate Burgers PDE, initial, boundary, and total losses."""
    equation = burgers_equation_residual(
        model=model,
        x=points.x_f,
        t=points.t_f,
        viscosity=config.viscosity,
    )
    pde = pde_residual_loss(
        equation.residual
    )

    initial_inputs = torch.cat(
        [points.x_i, points.t_i],
        dim=1,
    )
    initial_prediction = model(
        initial_inputs
    )
    initial_target = burgers_initial_condition(
        points.x_i
    )

    initial = initial_condition_loss(
        prediction=initial_prediction,
        target=initial_target,
    )

    left_prediction = model(
        torch.cat(
            [
                points.x_left,
                points.t_left,
            ],
            dim=1,
        )
    )
    right_prediction = model(
        torch.cat(
            [
                points.x_right,
                points.t_right,
            ],
            dim=1,
        )
    )

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


def trainable_parameter_count(
    model: nn.Module,
) -> int:
    """Count trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def quantum_parameter_count(
    model: nn.Module,
) -> int:
    """Count trainable circuit angles."""
    if not isinstance(
        model,
        HybridQuantumPINN,
    ):
        return 0

    return (
        model.quantum_layer
        .quantum_weights
        .numel()
    )


def quantum_gradient_norm(
    model: nn.Module,
) -> float | None:
    """Return the current quantum-parameter gradient norm."""
    if not isinstance(
        model,
        HybridQuantumPINN,
    ):
        return None

    gradient = (
        model.quantum_layer
        .quantum_weights
        .grad
    )

    if gradient is None:
        return 0.0

    return float(
        torch.linalg.vector_norm(
            gradient.detach()
        ).cpu()
    )


def copy_quantum_parameters(
    model: nn.Module,
) -> torch.Tensor | None:
    """Copy initial quantum parameters for an update check."""
    if not isinstance(
        model,
        HybridQuantumPINN,
    ):
        return None

    return (
        model.quantum_layer
        .quantum_weights
        .detach()
        .cpu()
        .clone()
    )


def quantum_parameter_change_norm(
    model: nn.Module,
    initial: torch.Tensor | None,
) -> float | None:
    """Measure the change in trainable circuit angles."""
    if (
        not isinstance(
            model,
            HybridQuantumPINN,
        )
        or initial is None
    ):
        return None

    return float(
        torch.linalg.vector_norm(
            model.quantum_layer
            .quantum_weights
            .detach()
            .cpu()
            - initial
        )
    )


def train_model(
    model: nn.Module,
    points: BurgersTrainingPoints,
    config: BurgersTrainingConfig,
) -> dict[str, Any]:
    """Train one model on the shared Burgers points."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    initial_quantum = (
        copy_quantum_parameters(model)
    )

    history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "initial": [],
        "boundary": [],
        "global_gradient": [],
        "quantum_gradient": [],
    }

    start = time.perf_counter()

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        losses = calculate_burgers_losses(
            model=model,
            points=points,
            config=config,
        )

        losses.total.backward()

        global_norm = global_gradient_norm(
            model
        )
        quantum_norm = quantum_gradient_norm(
            model
        )

        optimizer.step()

        history["total"].append(
            float(
                losses.total
                .detach()
                .cpu()
            )
        )
        history["pde"].append(
            float(
                losses.pde
                .detach()
                .cpu()
            )
        )
        history["initial"].append(
            float(
                losses.initial
                .detach()
                .cpu()
            )
        )
        history["boundary"].append(
            float(
                losses.boundary
                .detach()
                .cpu()
            )
        )
        history[
            "global_gradient"
        ].append(global_norm)

        if quantum_norm is not None:
            history[
                "quantum_gradient"
            ].append(quantum_norm)

        if (
            epoch == 1
            or epoch == config.epochs
            or epoch % config.log_every == 0
        ):
            quantum_text = (
                ""
                if quantum_norm is None
                else (
                    " | q_grad="
                    f"{quantum_norm:.6e}"
                )
            )

            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"total={history['total'][-1]:.6e} | "
                f"pde={history['pde'][-1]:.6e} | "
                f"ic={history['initial'][-1]:.6e} | "
                f"bc={history['boundary'][-1]:.6e} | "
                f"grad={global_norm:.6e}"
                f"{quantum_text}"
            )

    elapsed = (
        time.perf_counter() - start
    )

    if not math.isfinite(
        history["total"][-1]
    ):
        raise RuntimeError(
            "The final loss is not finite."
        )

    if (
        history["total"][-1]
        >= history["total"][0]
    ):
        raise RuntimeError(
            "The total loss did not decrease."
        )

    return {
        "history": history,
        "training_seconds": elapsed,
        "quantum_parameter_change_norm": (
            quantum_parameter_change_norm(
                model=model,
                initial=initial_quantum,
            )
        ),
    }


def load_reference(
    reference_path: Path,
) -> BurgersReferenceSolution:
    """Load the validated Cole-Hopf reference file."""
    with np.load(
        reference_path,
        allow_pickle=False,
    ) as data:
        return BurgersReferenceSolution(
            x=np.asarray(
                data["x"],
                dtype=np.float64,
            ),
            t=np.asarray(
                data["t"],
                dtype=np.float64,
            ),
            u=np.asarray(
                data["u"],
                dtype=np.float64,
            ),
            viscosity=float(
                data["viscosity"]
            ),
            quadrature_order=int(
                data["quadrature_order"]
            ),
            reference_method=str(
                data["reference_method"]
            ),
        )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    reference: BurgersReferenceSolution,
    bands: FrequencyBands,
) -> dict[str, Any]:
    """Evaluate global and spectral errors on the reference grid."""
    dtype = next(
        model.parameters()
    ).dtype

    x = torch.as_tensor(
        reference.x,
        dtype=dtype,
    )
    t = torch.as_tensor(
        reference.t,
        dtype=dtype,
    )

    x_grid, t_grid = torch.meshgrid(
        x,
        t,
        indexing="ij",
    )

    inputs = torch.stack(
        [
            x_grid.reshape(-1),
            t_grid.reshape(-1),
        ],
        dim=1,
    )

    # Chunked evaluation keeps the state-vector memory small.
    predictions = []

    for start_index in range(
        0,
        inputs.shape[0],
        4096,
    ):
        predictions.append(
            model(
                inputs[
                    start_index:
                    start_index + 4096
                ]
            ).cpu()
        )

    prediction = torch.cat(
        predictions,
        dim=0,
    ).reshape(
        reference.x.size,
        reference.t.size,
    ).numpy()

    error = prediction - reference.u

    relative_l2 = float(
        np.linalg.norm(error)
        / np.linalg.norm(reference.u)
    )
    mean_absolute_error = float(
        np.mean(
            np.abs(error)
        )
    )
    maximum_absolute_error = float(
        np.max(
            np.abs(error)
        )
    )

    spectral_metrics: dict[str, Any] = {}

    for requested_time in (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        time_index = int(
            np.argmin(
                np.abs(
                    reference.t
                    - requested_time
                )
            )
        )
        actual_time = float(
            reference.t[
                time_index
            ]
        )

        analysis = analyze_spatial_profile(
            x=reference.x,
            prediction=prediction[
                :,
                time_index,
            ],
            reference=reference.u[
                :,
                time_index,
            ],
            bands=bands,
            use_odd_extension=True,
        )

        spectral_metrics[
            f"{actual_time:.6f}"
        ] = {
            "dominant_reference_frequency": (
                analysis[
                    "dominant_reference_frequency"
                ]
            ),
            "dominant_prediction_frequency": (
                analysis[
                    "dominant_prediction_frequency"
                ]
            ),
            "spatial_relative_l2": (
                analysis[
                    "spatial_relative_l2"
                ]
            ),
            "bands": {
                band_name: {
                    "reference_energy_fraction": (
                        band_result
                        .reference_energy_fraction
                    ),
                    "prediction_energy_fraction": (
                        band_result
                        .prediction_energy_fraction
                    ),
                    (
                        "error_normalized_by_"
                        "total_reference_energy"
                    ): (
                        band_result
                        .error_normalized_by_total_reference_energy
                    ),
                    "relative_band_error": (
                        band_result
                        .relative_band_error
                    ),
                }
                for (
                    band_name,
                    band_result,
                ) in analysis[
                    "band_metrics"
                ].items()
            },
        }

    return {
        "relative_l2": relative_l2,
        "mean_absolute_error": (
            mean_absolute_error
        ),
        "maximum_absolute_error": (
            maximum_absolute_error
        ),
        "spectral_metrics": (
            spectral_metrics
        ),
        "prediction": prediction,
    }


def run_one_model(
    specification: ModelSpecification,
    seed: int,
    points: BurgersTrainingPoints,
    config: BurgersTrainingConfig,
    reference: BurgersReferenceSolution,
    bands: FrequencyBands,
    output_directory: Path,
) -> BurgersRunResult:
    """Initialize, train, evaluate, and save one model."""
    set_reproducible_seed(seed)
    model = specification.constructor()

    training = train_model(
        model=model,
        points=points,
        config=config,
    )
    evaluation = evaluate_model(
        model=model,
        reference=reference,
        bands=bands,
    )

    quantum_history = (
        training["history"][
            "quantum_gradient"
        ]
    )

    result = BurgersRunResult(
        model=specification.name,
        architecture=(
            specification.architecture
        ),
        seed=seed,
        trainable_parameters=(
            trainable_parameter_count(
                model
            )
        ),
        quantum_parameters=(
            quantum_parameter_count(
                model
            )
        ),
        initial_total_loss=(
            training["history"][
                "total"
            ][0]
        ),
        final_total_loss=(
            training["history"][
                "total"
            ][-1]
        ),
        final_pde_loss=(
            training["history"][
                "pde"
            ][-1]
        ),
        final_initial_loss=(
            training["history"][
                "initial"
            ][-1]
        ),
        final_boundary_loss=(
            training["history"][
                "boundary"
            ][-1]
        ),
        final_global_gradient_norm=(
            training["history"][
                "global_gradient"
            ][-1]
        ),
        final_quantum_gradient_norm=(
            quantum_history[-1]
            if quantum_history
            else None
        ),
        quantum_parameter_change_norm=(
            training[
                "quantum_parameter_change_norm"
            ]
        ),
        relative_l2=(
            evaluation["relative_l2"]
        ),
        mean_absolute_error=(
            evaluation[
                "mean_absolute_error"
            ]
        ),
        maximum_absolute_error=(
            evaluation[
                "maximum_absolute_error"
            ]
        ),
        training_seconds=(
            training["training_seconds"]
        ),
        spectral_metrics=(
            evaluation["spectral_metrics"]
        ),
    )

    run_directory = (
        output_directory
        / f"seed_{seed}"
        / specification.name
    )
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        run_directory
        / "model_state.pt",
    )
    np.save(
        run_directory
        / "prediction.npy",
        evaluation["prediction"],
    )

    with (
        run_directory
        / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training["history"],
            file,
            indent=2,
        )

    with (
        run_directory
        / "run_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return result


def aggregate_scalar(
    runs: list[BurgersRunResult],
    field: str,
) -> dict[str, float]:
    """Return mean and sample standard deviation."""
    values = [
        float(
            getattr(
                run,
                field,
            )
        )
        for run in runs
    ]

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def aggregate_spectral_error(
    runs: list[BurgersRunResult],
    time_key: str,
    band_name: str,
) -> dict[str, float]:
    """Aggregate one stable spectral band error."""
    metric_name = (
        "error_normalized_by_total_reference_energy"
    )

    values = [
        float(
            run
            .spectral_metrics[
                time_key
            ]["bands"][
                band_name
            ][metric_name]
        )
        for run in runs
    ]

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def summarize_results(
    all_runs: list[BurgersRunResult],
) -> dict[str, Any]:
    """Create model-wise multi-seed summaries."""
    summary: dict[str, Any] = {}

    for specification in model_specifications():
        selected = [
            run
            for run in all_runs
            if run.model
            == specification.name
        ]

        model_summary: dict[str, Any] = {
            "architecture": (
                specification.architecture
            ),
            "number_of_seeds": (
                len(selected)
            ),
            "trainable_parameters": (
                selected[0]
                .trainable_parameters
            ),
            "quantum_parameters": (
                selected[0]
                .quantum_parameters
            ),
            "relative_l2": (
                aggregate_scalar(
                    selected,
                    "relative_l2",
                )
            ),
            "mean_absolute_error": (
                aggregate_scalar(
                    selected,
                    "mean_absolute_error",
                )
            ),
            "maximum_absolute_error": (
                aggregate_scalar(
                    selected,
                    "maximum_absolute_error",
                )
            ),
            "final_total_loss": (
                aggregate_scalar(
                    selected,
                    "final_total_loss",
                )
            ),
            "final_pde_loss": (
                aggregate_scalar(
                    selected,
                    "final_pde_loss",
                )
            ),
            "final_initial_loss": (
                aggregate_scalar(
                    selected,
                    "final_initial_loss",
                )
            ),
            "final_boundary_loss": (
                aggregate_scalar(
                    selected,
                    "final_boundary_loss",
                )
            ),
            "training_seconds": (
                aggregate_scalar(
                    selected,
                    "training_seconds",
                )
            ),
            "spectral_errors": {},
        }

        for time_key in (
            "0.000000",
            "0.250000",
            "0.500000",
            "0.750000",
            "1.000000",
        ):
            model_summary[
                "spectral_errors"
            ][time_key] = {
                band_name: (
                    aggregate_spectral_error(
                        selected,
                        time_key,
                        band_name,
                    )
                )
                for band_name in (
                    "low",
                    "middle",
                    "high",
                )
            }

        summary[
            specification.name
        ] = model_summary

    return summary


def save_scalar_csv(
    runs: list[BurgersRunResult],
    output_path: Path,
) -> None:
    """Save run-level scalar metrics."""
    rows = []

    for run in runs:
        row = asdict(run)
        row.pop(
            "spectral_metrics"
        )
        rows.append(row)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figures(
    runs: list[BurgersRunResult],
    summary: dict[str, Any],
    reference: BurgersReferenceSolution,
    output_directory: Path,
) -> None:
    """Save independent scientific comparison figures."""
    model_names = [
        specification.name
        for specification
        in model_specifications()
    ]
    seeds = sorted(
        {
            run.seed
            for run in runs
        }
    )

    # Relative L2 by seed.
    figure = plt.figure(
        figsize=(8, 5)
    )
    axis = figure.add_subplot(111)

    positions = np.arange(
        len(seeds)
    )
    width = 0.35

    for index, model_name in enumerate(
        model_names
    ):
        selected = {
            run.seed: run
            for run in runs
            if run.model == model_name
        }
        offset = (
            index
            - (
                len(model_names) - 1
            ) / 2
        ) * width

        axis.bar(
            positions + offset,
            [
                selected[
                    seed
                ].relative_l2
                for seed in seeds
            ],
            width=width,
            label=model_name,
        )

    axis.set_xticks(
        positions,
        [
            str(seed)
            for seed in seeds
        ],
    )
    axis.set_xlabel(
        "Random seed"
    )
    axis.set_ylabel(
        "Relative L2 error"
    )
    axis.set_title(
        "Burgers comparison by seed"
    )
    axis.legend()
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "relative_l2_by_seed.png",
        dpi=200,
    )
    plt.close(figure)

    # Mean relative L2.
    figure = plt.figure(
        figsize=(8, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        model_names,
        [
            summary[name][
                "relative_l2"
            ]["mean"]
            for name in model_names
        ],
        yerr=[
            summary[name][
                "relative_l2"
            ]["std"]
            for name in model_names
        ],
        capsize=5,
    )
    axis.set_ylabel(
        "Relative L2 error"
    )
    axis.set_title(
        "Mean Burgers error across seeds"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "relative_l2_mean_std.png",
        dpi=200,
    )
    plt.close(figure)

    # Spectral band error at t=0.5.
    figure = plt.figure(
        figsize=(8, 5)
    )
    axis = figure.add_subplot(111)

    band_names = [
        "low",
        "middle",
        "high",
    ]
    positions = np.arange(
        len(band_names)
    )

    for index, model_name in enumerate(
        model_names
    ):
        offset = (
            index
            - (
                len(model_names) - 1
            ) / 2
        ) * width

        values = summary[
            model_name
        ]["spectral_errors"][
            "0.500000"
        ]

        axis.bar(
            positions + offset,
            [
                values[band][
                    "mean"
                ]
                for band in band_names
            ],
            width=width,
            yerr=[
                values[band][
                    "std"
                ]
                for band in band_names
            ],
            capsize=4,
            label=model_name,
        )

    axis.set_xticks(
        positions,
        band_names,
    )
    axis.set_xlabel(
        "Spatial-frequency band"
    )
    axis.set_ylabel(
        "Error / total reference spectral energy"
    )
    axis.set_title(
        "Burgers spectral errors at t=0.5"
    )
    axis.legend()
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "spectral_errors_t_0p5.png",
        dpi=200,
    )
    plt.close(figure)

    # Solution slices for seed 22.
    selected_seed = (
        22
        if 22 in seeds
        else seeds[0]
    )

    for time_value in (
        0.0,
        0.5,
        1.0,
    ):
        time_index = int(
            np.argmin(
                np.abs(
                    reference.t
                    - time_value
                )
            )
        )

        figure = plt.figure(
            figsize=(8, 5)
        )
        axis = figure.add_subplot(111)

        axis.plot(
            reference.x,
            reference.u[
                :,
                time_index,
            ],
            label="Cole-Hopf reference",
            linewidth=2,
        )

        for model_name in model_names:
            prediction_path = (
                output_directory
                / f"seed_{selected_seed}"
                / model_name
                / "prediction.npy"
            )
            prediction = np.load(
                prediction_path
            )

            axis.plot(
                reference.x,
                prediction[
                    :,
                    time_index,
                ],
                label=model_name,
            )

        axis.set_xlabel("x")
        axis.set_ylabel("u(x,t)")
        axis.set_title(
            "Burgers solution slice: "
            f"t={reference.t[time_index]:.2f}, "
            f"seed={selected_seed}"
        )
        axis.legend()
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()
        figure.savefig(
            output_directory
            / (
                "solution_slice_"
                f"t_{reference.t[time_index]:.2f}"
                ".png"
            ),
            dpi=200,
        )
        plt.close(figure)


def write_readme(
    summary: dict[str, Any],
    config: BurgersTrainingConfig,
    seeds: list[int],
    output_path: Path,
) -> None:
    """Write an honest report for this preliminary pilot."""
    classical = summary[
        "classical_bottleneck"
    ]
    quantum = summary[
        "four_qubit_qapinn"
    ]

    lines = [
        "# Matched Burgers PINN pilot",
        "",
        "## Benchmark",
        "",
        (
            "- PDE: u_t + u*u_x = viscosity*u_xx"
        ),
        (
            "- viscosity = 0.01/pi"
        ),
        (
            "- Initial condition: u(x,0) = -sin(pi*x)"
        ),
        (
            "- Boundary conditions: u(-1,t)=u(1,t)=0"
        ),
        (
            "- Reference: Cole-Hopf integral evaluated with "
            "120-point Gauss-Hermite quadrature"
        ),
        "",
        "## Controlled training",
        "",
        f"- Seeds: {seeds}",
        f"- Epochs: {config.epochs}",
        f"- Collocation points: {config.n_collocation}",
        f"- Initial points: {config.n_initial}",
        f"- Boundary points: {config.n_boundary}",
        (
            "- Both models received exactly the same sampled "
            "coordinates within each seed."
        ),
        "",
        "## Mean relative L2 error",
        "",
        (
            "- Classical bottleneck: "
            f"{classical['relative_l2']['mean']:.6f} ± "
            f"{classical['relative_l2']['std']:.6f}"
        ),
        (
            "- Four-qubit QAPINN: "
            f"{quantum['relative_l2']['mean']:.6f} ± "
            f"{quantum['relative_l2']['std']:.6f}"
        ),
        "",
        "## Interpretation constraint",
        "",
        (
            "This is a short pilot with small point counts. "
            "It tests whether the complete nonlinear workflow runs; "
            "it is not a final accuracy benchmark and does not establish "
            "quantum advantage or inferiority."
        ),
        (
            "The quantum model uses an ideal differentiable state-vector "
            "simulator, not quantum hardware."
        ),
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched classical-bottleneck versus "
            "four-qubit QAPINN Burgers pilot."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "data/burgers_reference.npz"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/fair_classical_quantum_burgers"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    reference = load_reference(
        arguments.reference
    )

    config = BurgersTrainingConfig(
        viscosity=reference.viscosity,
        epochs=arguments.epochs,
        learning_rate=1.0e-3,
        n_collocation=128,
        n_initial=32,
        n_boundary=32,
        log_every=arguments.epochs,
    )

    bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_runs: list[
        BurgersRunResult
    ] = []

    for seed in arguments.seeds:
        set_reproducible_seed(seed)

        shared_points = (
            sample_burgers_training_points(
                config=config,
                dtype=torch.float32,
            )
        )

        for specification in model_specifications():
            print(
                "\n========================================"
            )
            print(
                f"Seed {seed} | "
                f"{specification.name}"
            )
            print(
                "========================================"
            )

            run = run_one_model(
                specification=specification,
                seed=seed,
                points=shared_points,
                config=config,
                reference=reference,
                bands=bands,
                output_directory=(
                    arguments.output_directory
                ),
            )

            all_runs.append(run)

            print(
                f"Completed {specification.name}, "
                f"seed {seed}: "
                f"relative L2={run.relative_l2:.6f}, "
                f"time={run.training_seconds:.2f}s"
            )

    summary = summarize_results(
        all_runs
    )

    with (
        arguments.output_directory
        / "runs.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                asdict(run)
                for run in all_runs
            ],
            file,
            indent=2,
        )

    with (
        arguments.output_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    with (
        arguments.output_directory
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "training": (
                    asdict(config)
                ),
                "seeds": (
                    arguments.seeds
                ),
                "frequency_bands": (
                    asdict(bands)
                ),
                "reference_path": (
                    str(arguments.reference)
                ),
                "reference_method": (
                    reference.reference_method
                ),
                "shared_points_within_seed": (
                    True
                ),
            },
            file,
            indent=2,
        )

    save_scalar_csv(
        runs=all_runs,
        output_path=(
            arguments.output_directory
            / "runs.csv"
        ),
    )

    save_figures(
        runs=all_runs,
        summary=summary,
        reference=reference,
        output_directory=(
            arguments.output_directory
        ),
    )

    write_readme(
        summary=summary,
        config=config,
        seeds=arguments.seeds,
        output_path=(
            arguments.output_directory
            / "README.md"
        ),
    )

    print(
        "\nAggregate Burgers comparison"
    )

    for model_name, values in (
        summary.items()
    ):
        print(
            f"{model_name:24s} | "
            f"params="
            f"{values['trainable_parameters']:3d} | "
            f"relative L2="
            f"{values['relative_l2']['mean']:.6f} ± "
            f"{values['relative_l2']['std']:.6f} | "
            f"time="
            f"{values['training_seconds']['mean']:.2f}s"
        )

    print(
        "\nSaved outputs to: "
        f"{arguments.output_directory}"
    )
