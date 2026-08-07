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

from classical_bottleneck import ClassicalBottleneckPINN
from quantum_layer import HybridQuantumPINN
from spectra import FrequencyBands, analyze_spatial_profile
from train_heat import (
    HeatTrainingConfig,
    calculate_heat_losses,
    global_gradient_norm,
    sample_heat_training_points,
    set_reproducible_seed,
)
from heat import heat_exact_solution


@dataclass(frozen=True)
class AblationSpecification:
    """Definition of one controlled model variant."""

    name: str
    architecture: str
    model_family: str
    constructor: Callable[[], nn.Module]
    freeze_quantum_layer: bool = False


@dataclass
class AblationRun:
    """Complete scalar and spectral results from one run."""

    model: str
    architecture: str
    model_family: str
    seed: int

    entanglement: str | None
    quantum_frozen: bool

    trainable_parameters: int
    total_parameters: int
    total_quantum_parameters: int
    trainable_quantum_parameters: int

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


def ablation_specifications() -> list[AblationSpecification]:
    """Return the classical reference and three quantum ablations."""
    return [
        AblationSpecification(
            name="classical_bottleneck",
            architecture="2 -> 8 -> 4 -> 8 -> 1",
            model_family="classical",
            constructor=lambda: ClassicalBottleneckPINN(
                in_features=2,
                classical_width=8,
                bottleneck_dimension=4,
                out_features=1,
            ),
        ),
        AblationSpecification(
            name="qapinn_no_entanglement",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "4-qubit VQC(depth=2, no CNOT) -> "
                "4 <Z> -> 8 -> 1"
            ),
            model_family="quantum_assisted",
            constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="none",
                out_features=1,
            ),
        ),
        AblationSpecification(
            name="qapinn_ring_entanglement",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "4-qubit VQC(depth=2, ring CNOTs) -> "
                "4 <Z> -> 8 -> 1"
            ),
            model_family="quantum_assisted",
            constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
        AblationSpecification(
            name="qapinn_ring_frozen",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "frozen 4-qubit VQC(depth=2, ring CNOTs) -> "
                "4 <Z> -> 8 -> 1"
            ),
            model_family="quantum_assisted_frozen",
            constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
            freeze_quantum_layer=True,
        ),
    ]


def total_parameter_count(model: nn.Module) -> int:
    """Count every scalar parameter, including frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    """Count only parameters updated by the optimizer."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def total_quantum_parameter_count(model: nn.Module) -> int:
    """Count all circuit parameters, trainable or frozen."""
    if not isinstance(model, HybridQuantumPINN):
        return 0

    return model.quantum_layer.quantum_weights.numel()


def trainable_quantum_parameter_count(model: nn.Module) -> int:
    """Count circuit parameters that remain trainable."""
    if not isinstance(model, HybridQuantumPINN):
        return 0

    parameter = model.quantum_layer.quantum_weights
    return parameter.numel() if parameter.requires_grad else 0


def freeze_quantum_parameters(model: nn.Module) -> None:
    """Freeze only the variational circuit angles."""
    if not isinstance(model, HybridQuantumPINN):
        raise TypeError(
            "Only HybridQuantumPINN contains quantum parameters."
        )

    model.quantum_layer.quantum_weights.requires_grad_(False)


def copy_quantum_parameters(
    model: nn.Module,
) -> torch.Tensor | None:
    """Save initial quantum angles for change-norm analysis."""
    if not isinstance(model, HybridQuantumPINN):
        return None

    return (
        model.quantum_layer.quantum_weights
        .detach()
        .cpu()
        .clone()
    )


def quantum_parameter_change_norm(
    model: nn.Module,
    initial: torch.Tensor | None,
) -> float | None:
    """Measure the Euclidean change of the circuit angles."""
    if (
        not isinstance(model, HybridQuantumPINN)
        or initial is None
    ):
        return None

    return float(
        torch.linalg.vector_norm(
            model.quantum_layer.quantum_weights
            .detach()
            .cpu()
            - initial
        )
    )


def current_quantum_gradient_norm(
    model: nn.Module,
) -> float | None:
    """Return the circuit-gradient norm when it is trainable."""
    if not isinstance(model, HybridQuantumPINN):
        return None

    parameter = model.quantum_layer.quantum_weights

    if not parameter.requires_grad:
        return None

    if parameter.grad is None:
        return 0.0

    return float(
        torch.linalg.vector_norm(
            parameter.grad.detach()
        ).cpu()
    )


def train_model(
    model: nn.Module,
    points,
    config: HeatTrainingConfig,
) -> dict[str, Any]:
    """Train one model on the shared points for one seed."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
    )

    initial_quantum = copy_quantum_parameters(model)

    history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "initial": [],
        "boundary": [],
        "global_gradient": [],
        "quantum_gradient": [],
    }

    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        losses = calculate_heat_losses(
            model=model,
            points=points,
            config=config,
        )

        losses.total.backward()

        global_norm = global_gradient_norm(model)
        quantum_norm = current_quantum_gradient_norm(model)

        optimizer.step()

        history["total"].append(
            float(losses.total.detach().cpu())
        )
        history["pde"].append(
            float(losses.pde.detach().cpu())
        )
        history["initial"].append(
            float(losses.initial.detach().cpu())
        )
        history["boundary"].append(
            float(losses.boundary.detach().cpu())
        )
        history["global_gradient"].append(global_norm)

        if quantum_norm is not None:
            history["quantum_gradient"].append(
                quantum_norm
            )

        if (
            epoch == 1
            or epoch == config.epochs
            or epoch % config.log_every == 0
        ):
            quantum_text = (
                ""
                if quantum_norm is None
                else f" | q_grad={quantum_norm:.6e}"
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

    elapsed = time.perf_counter() - start

    if not math.isfinite(history["total"][-1]):
        raise RuntimeError("The final loss is not finite.")

    if history["total"][-1] >= history["total"][0]:
        raise RuntimeError(
            "The total loss did not decrease during training."
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


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    alpha: float,
    bands: FrequencyBands,
    n_x: int = 201,
    n_t: int = 101,
) -> dict[str, Any]:
    """Evaluate global and spatial-frequency errors."""
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
        torch.cat(
            [x_column, t_column],
            dim=1,
        )
    )
    reference = heat_exact_solution(
        x=x_column,
        t=t_column,
        alpha=alpha,
    )

    error = prediction - reference

    relative_l2 = float(
        (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(reference)
        ).cpu()
    )
    mae = float(
        torch.mean(torch.abs(error)).cpu()
    )
    maximum_error = float(
        torch.max(torch.abs(error)).cpu()
    )

    prediction_grid = (
        prediction.reshape(n_x, n_t)
        .cpu()
        .numpy()
    )
    reference_grid = (
        reference.reshape(n_x, n_t)
        .cpu()
        .numpy()
    )
    x_values = x.cpu().numpy()
    t_values = t.cpu().numpy()

    spectral_metrics: dict[str, Any] = {}

    for requested_time in (0.0, 0.5, 1.0):
        time_index = int(
            np.argmin(
                np.abs(
                    t_values - requested_time
                )
            )
        )
        actual_time = float(
            t_values[time_index]
        )

        analysis = analyze_spatial_profile(
            x=x_values,
            prediction=prediction_grid[
                :,
                time_index,
            ],
            reference=reference_grid[
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
                analysis["spatial_relative_l2"]
            ),
            "bands": {
                band_name: {
                    (
                        "error_normalized_by_"
                        "total_reference_energy"
                    ): (
                        band_result
                        .error_normalized_by_total_reference_energy
                    ),
                    "prediction_energy_fraction": (
                        band_result
                        .prediction_energy_fraction
                    ),
                    "reference_energy_fraction": (
                        band_result
                        .reference_energy_fraction
                    ),
                }
                for band_name, band_result
                in analysis[
                    "band_metrics"
                ].items()
            },
        }

    return {
        "relative_l2": relative_l2,
        "mean_absolute_error": mae,
        "maximum_absolute_error": maximum_error,
        "spectral_metrics": spectral_metrics,
    }


def run_one_ablation(
    specification: AblationSpecification,
    seed: int,
    shared_points,
    config: HeatTrainingConfig,
    bands: FrequencyBands,
    output_directory: Path,
) -> AblationRun:
    """Run and save one controlled ablation."""
    set_reproducible_seed(seed)
    model = specification.constructor()

    if specification.freeze_quantum_layer:
        freeze_quantum_parameters(model)

    training = train_model(
        model=model,
        points=shared_points,
        config=config,
    )
    evaluation = evaluate_model(
        model=model,
        alpha=config.alpha,
        bands=bands,
    )

    entanglement = (
        model.entanglement
        if isinstance(model, HybridQuantumPINN)
        else None
    )

    final_quantum_gradient = (
        training["history"]["quantum_gradient"][-1]
        if training["history"]["quantum_gradient"]
        else None
    )

    result = AblationRun(
        model=specification.name,
        architecture=specification.architecture,
        model_family=specification.model_family,
        seed=seed,
        entanglement=entanglement,
        quantum_frozen=(
            specification.freeze_quantum_layer
        ),
        trainable_parameters=(
            trainable_parameter_count(model)
        ),
        total_parameters=(
            total_parameter_count(model)
        ),
        total_quantum_parameters=(
            total_quantum_parameter_count(model)
        ),
        trainable_quantum_parameters=(
            trainable_quantum_parameter_count(model)
        ),
        initial_total_loss=(
            training["history"]["total"][0]
        ),
        final_total_loss=(
            training["history"]["total"][-1]
        ),
        final_pde_loss=(
            training["history"]["pde"][-1]
        ),
        final_initial_loss=(
            training["history"]["initial"][-1]
        ),
        final_boundary_loss=(
            training["history"]["boundary"][-1]
        ),
        final_global_gradient_norm=(
            training[
                "history"
            ]["global_gradient"][-1]
        ),
        final_quantum_gradient_norm=(
            final_quantum_gradient
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
            evaluation["mean_absolute_error"]
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
        run_directory / "model_state.pt",
    )

    with (
        run_directory / "history.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            training["history"],
            file,
            indent=2,
        )

    with (
        run_directory / "run_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return result


def aggregate(
    selected: list[AblationRun],
    field: str,
) -> dict[str, float]:
    """Aggregate one scalar run metric."""
    values = [
        float(getattr(run, field))
        for run in selected
    ]

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def aggregate_optional(
    selected: list[AblationRun],
    field: str,
) -> dict[str, float] | None:
    """Aggregate an optional scalar metric."""
    values = [
        float(value)
        for run in selected
        if (
            value := getattr(run, field)
        ) is not None
    ]

    if not values:
        return None

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def aggregate_spectral_error(
    selected: list[AblationRun],
    time_key: str,
    band: str,
) -> dict[str, float]:
    """Aggregate a stable band-normalized spectral error."""
    metric = (
        "error_normalized_by_total_reference_energy"
    )

    values = [
        float(
            run
            .spectral_metrics[time_key]
            ["bands"][band][metric]
        )
        for run in selected
    ]

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def summarize_runs(
    runs: list[AblationRun],
) -> dict[str, Any]:
    """Create model-wise multi-seed summaries."""
    summary: dict[str, Any] = {}

    for specification in ablation_specifications():
        selected = [
            run
            for run in runs
            if run.model == specification.name
        ]

        if not selected:
            raise ValueError(
                f"No runs found for {specification.name}."
            )

        model_summary: dict[str, Any] = {
            "architecture": specification.architecture,
            "model_family": specification.model_family,
            "number_of_seeds": len(selected),
            "trainable_parameters": (
                selected[0].trainable_parameters
            ),
            "total_parameters": (
                selected[0].total_parameters
            ),
            "total_quantum_parameters": (
                selected[0].total_quantum_parameters
            ),
            "trainable_quantum_parameters": (
                selected[0]
                .trainable_quantum_parameters
            ),
            "relative_l2": aggregate(
                selected,
                "relative_l2",
            ),
            "mean_absolute_error": aggregate(
                selected,
                "mean_absolute_error",
            ),
            "maximum_absolute_error": aggregate(
                selected,
                "maximum_absolute_error",
            ),
            "final_total_loss": aggregate(
                selected,
                "final_total_loss",
            ),
            "final_pde_loss": aggregate(
                selected,
                "final_pde_loss",
            ),
            "final_initial_loss": aggregate(
                selected,
                "final_initial_loss",
            ),
            "final_boundary_loss": aggregate(
                selected,
                "final_boundary_loss",
            ),
            "training_seconds": aggregate(
                selected,
                "training_seconds",
            ),
            "final_quantum_gradient_norm": (
                aggregate_optional(
                    selected,
                    "final_quantum_gradient_norm",
                )
            ),
            "quantum_parameter_change_norm": (
                aggregate_optional(
                    selected,
                    "quantum_parameter_change_norm",
                )
            ),
            "spectral_errors": {},
        }

        for time_key in (
            "0.000000",
            "0.500000",
            "1.000000",
        ):
            model_summary[
                "spectral_errors"
            ][time_key] = {
                band: aggregate_spectral_error(
                    selected=selected,
                    time_key=time_key,
                    band=band,
                )
                for band in (
                    "low",
                    "middle",
                    "high",
                )
            }

        summary[
            specification.name
        ] = model_summary

    return summary


def save_run_csv(
    runs: list[AblationRun],
    output_path: Path,
) -> None:
    """Save scalar run values without nested spectral dictionaries."""
    rows = []

    for run in runs:
        row = asdict(run)
        row.pop("spectral_metrics")
        rows.append(row)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figures(
    runs: list[AblationRun],
    summary: dict[str, Any],
    output_directory: Path,
) -> None:
    """Create independent figures for performance and ablations."""
    model_names = [
        specification.name
        for specification in ablation_specifications()
    ]
    seeds = sorted(
        {run.seed for run in runs}
    )

    # Relative L2 by seed.
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)

    x_positions = np.arange(len(seeds))
    width = 0.19

    for index, model_name in enumerate(model_names):
        model_results = {
            run.seed: run
            for run in runs
            if run.model == model_name
        }
        offset = (
            index
            - (len(model_names) - 1) / 2
        ) * width

        axis.bar(
            x_positions + offset,
            [
                model_results[seed].relative_l2
                for seed in seeds
            ],
            width=width,
            label=model_name,
        )

    axis.set_xticks(
        x_positions,
        [str(seed) for seed in seeds],
    )
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Relative L2 error")
    axis.set_title(
        "Heat PINN quantum-ablation results by seed"
    )
    axis.legend(fontsize=8)
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
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)

    means = [
        summary[name]["relative_l2"]["mean"]
        for name in model_names
    ]
    deviations = [
        summary[name]["relative_l2"]["std"]
        for name in model_names
    ]

    axis.bar(
        model_names,
        means,
        yerr=deviations,
        capsize=5,
    )
    axis.set_ylabel("Relative L2 error")
    axis.set_title(
        "Mean Heat error across seeds"
    )
    axis.tick_params(
        axis="x",
        rotation=15,
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

    # Spectral errors at t=0.5.
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)

    band_names = [
        "low",
        "middle",
        "high",
    ]
    x_positions = np.arange(
        len(band_names)
    )

    for index, model_name in enumerate(model_names):
        offset = (
            index
            - (len(model_names) - 1) / 2
        ) * width

        model_spectra = summary[
            model_name
        ]["spectral_errors"]["0.500000"]

        axis.bar(
            x_positions + offset,
            [
                model_spectra[band]["mean"]
                for band in band_names
            ],
            width=width,
            yerr=[
                model_spectra[band]["std"]
                for band in band_names
            ],
            capsize=3,
            label=model_name,
        )

    axis.set_xticks(
        x_positions,
        band_names,
    )
    axis.set_xlabel("Spatial-frequency band")
    axis.set_ylabel(
        "Error / total exact spectral energy"
    )
    axis.set_title(
        "Quantum-ablation spectral errors at t=0.5"
    )
    axis.legend(fontsize=8)
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

    # Training time comparison.
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)

    time_means = [
        summary[name]["training_seconds"]["mean"]
        for name in model_names
    ]
    time_deviations = [
        summary[name]["training_seconds"]["std"]
        for name in model_names
    ]

    axis.bar(
        model_names,
        time_means,
        yerr=time_deviations,
        capsize=5,
    )
    axis.set_ylabel("Training time (seconds)")
    axis.set_title(
        "Training cost of the quantum ablations"
    )
    axis.tick_params(
        axis="x",
        rotation=15,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_directory
        / "training_time_mean_std.png",
        dpi=200,
    )
    plt.close(figure)


def write_readme(
    summary: dict[str, Any],
    config: HeatTrainingConfig,
    seeds: list[int],
    output_path: Path,
) -> None:
    """Write a concise, honest scientific interpretation."""
    lines = [
        "# Quantum-layer ablation study for the Heat PINN",
        "",
        "## Controlled conditions",
        "",
        f"- Seeds: {seeds}",
        f"- Epochs: {config.epochs}",
        f"- Collocation points: {config.n_collocation}",
        f"- Initial points: {config.n_initial}",
        f"- Boundary points: {config.n_boundary}",
        (
            "- Within each seed, every model received exactly the "
            "same collocation, initial, and boundary coordinates."
        ),
        "",
        "## Mean relative L2 error",
        "",
    ]

    for specification in ablation_specifications():
        values = summary[
            specification.name
        ]["relative_l2"]

        lines.append(
            "- "
            f"{specification.name}: "
            f"{values['mean']:.6f} ± "
            f"{values['std']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            (
                "- Comparing no-entanglement and ring-entanglement "
                "tests whether CNOT entanglement improved the pilot."
            ),
            (
                "- Comparing trainable and frozen ring circuits tests "
                "whether learned circuit angles added value beyond a "
                "fixed nonlinear quantum feature map."
            ),
            (
                "- The frozen circuit still performs quantum-state "
                "evolution, but its 16 circuit angles are not updated."
            ),
            (
                "- This is an ideal state-vector simulation, not "
                "hardware execution."
            ),
            (
                "- Three seeds and a short Heat benchmark are not enough "
                "to establish statistical significance or quantum advantage."
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare classical, no-entanglement, entangling, "
            "and frozen quantum Heat PINNs."
        )
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
        default=Path("outputs/quantum_ablation_heat"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    config = HeatTrainingConfig(
        alpha=0.1,
        epochs=arguments.epochs,
        learning_rate=1.0e-3,
        n_collocation=128,
        n_initial=32,
        n_boundary=32,
        pde_weight=1.0,
        initial_weight=1.0,
        boundary_weight=1.0,
        seed=arguments.seeds[0],
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

    runs: list[AblationRun] = []

    for seed in arguments.seeds:
        # Sample once so all variants see identical coordinates.
        set_reproducible_seed(seed)

        shared_points = sample_heat_training_points(
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        for specification in ablation_specifications():
            print(
                "\n=============================================="
            )
            print(
                f"Seed {seed} | {specification.name}"
            )
            print(
                "=============================================="
            )

            run = run_one_ablation(
                specification=specification,
                seed=seed,
                shared_points=shared_points,
                config=config,
                bands=bands,
                output_directory=(
                    arguments.output_directory
                ),
            )

            runs.append(run)

            print(
                f"Completed {specification.name}, seed {seed}: "
                f"relative L2={run.relative_l2:.6f}, "
                f"time={run.training_seconds:.2f}s"
            )

    summary = summarize_runs(runs)

    with (
        arguments.output_directory
        / "runs.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            [asdict(run) for run in runs],
            file,
            indent=2,
        )

    with (
        arguments.output_directory
        / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    with (
        arguments.output_directory
        / "config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training": asdict(config),
                "seeds": arguments.seeds,
                "frequency_bands": asdict(bands),
                "shared_points_within_seed": True,
                "backend": (
                    "ideal differentiable PyTorch "
                    "state-vector simulator"
                ),
            },
            file,
            indent=2,
        )

    save_run_csv(
        runs=runs,
        output_path=(
            arguments.output_directory
            / "runs.csv"
        ),
    )

    save_figures(
        runs=runs,
        summary=summary,
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

    print("\nAggregate quantum-ablation results")

    for model_name, values in summary.items():
        print(
            f"{model_name:28s} | "
            f"trainable={values['trainable_parameters']:3d} | "
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
