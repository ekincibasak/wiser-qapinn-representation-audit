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

from classical_bottleneck import (
    ClassicalBottleneckPINN,
    count_trainable_parameters as count_classical_parameters,
)
from quantum_layer import (
    HybridQuantumPINN,
    count_trainable_parameters as count_quantum_model_parameters,
    quantum_parameter_count,
)
from spectra import (
    FrequencyBands,
    analyze_spatial_profile,
)
from train_heat import (
    HeatTrainingConfig,
    calculate_heat_losses,
    global_gradient_norm,
    sample_heat_training_points,
    set_reproducible_seed,
)
from heat import heat_exact_solution


@dataclass(frozen=True)
class ModelSpecification:
    """Definition of one model in the controlled comparison."""

    name: str
    architecture: str
    model_family: str
    constructor: Callable[[], nn.Module]


@dataclass
class FairRunResult:
    """Results for one model trained on one shared sampled data set."""

    model: str
    architecture: str
    model_family: str
    seed: int

    trainable_parameters: int
    quantum_parameters: int

    epochs: int
    n_collocation: int
    n_initial: int
    n_boundary: int

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
            model_family="classical",
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
                "4-qubit VQC(depth=2, ring) -> 4 <Z> -> 8 -> 1"
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
    ]


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters for either supported model."""
    if isinstance(model, HybridQuantumPINN):
        return count_quantum_model_parameters(model)

    return count_classical_parameters(model)


def quantum_gradient_norm(
    model: nn.Module,
) -> float | None:
    """Return the norm of quantum-parameter gradients, if present."""
    if not isinstance(model, HybridQuantumPINN):
        return None

    gradient = model.quantum_layer.quantum_weights.grad

    if gradient is None:
        return 0.0

    return float(
        torch.linalg.vector_norm(
            gradient.detach()
        ).cpu()
    )


def clone_quantum_parameters(
    model: nn.Module,
) -> torch.Tensor | None:
    """Copy the initial quantum parameters when the model is a QAPINN."""
    if not isinstance(model, HybridQuantumPINN):
        return None

    return (
        model.quantum_layer.quantum_weights
        .detach()
        .cpu()
        .clone()
    )


def quantum_change_norm(
    model: nn.Module,
    initial_quantum_parameters: torch.Tensor | None,
) -> float | None:
    """Measure how far the trainable circuit angles moved."""
    if (
        not isinstance(model, HybridQuantumPINN)
        or initial_quantum_parameters is None
    ):
        return None

    return float(
        torch.linalg.vector_norm(
            model.quantum_layer.quantum_weights
            .detach()
            .cpu()
            - initial_quantum_parameters
        )
    )


def train_on_shared_points(
    model: nn.Module,
    points,
    config: HeatTrainingConfig,
) -> dict[str, Any]:
    """Train one model on an already sampled, shared point set."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    initial_quantum_parameters = clone_quantum_parameters(
        model
    )

    history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "initial": [],
        "boundary": [],
        "global_gradient": [],
        "quantum_gradient": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        losses = calculate_heat_losses(
            model=model,
            points=points,
            config=config,
        )

        losses.total.backward()

        current_global_gradient = global_gradient_norm(
            model
        )
        current_quantum_gradient = quantum_gradient_norm(
            model
        )

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
        history["global_gradient"].append(
            current_global_gradient
        )

        if current_quantum_gradient is not None:
            history["quantum_gradient"].append(
                current_quantum_gradient
            )

        if (
            epoch == 1
            or epoch == config.epochs
            or epoch % config.log_every == 0
        ):
            quantum_text = (
                ""
                if current_quantum_gradient is None
                else (
                    " | q_grad="
                    f"{current_quantum_gradient:.6e}"
                )
            )

            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"total={history['total'][-1]:.6e} | "
                f"pde={history['pde'][-1]:.6e} | "
                f"ic={history['initial'][-1]:.6e} | "
                f"bc={history['boundary'][-1]:.6e} | "
                f"grad={current_global_gradient:.6e}"
                f"{quantum_text}"
            )

    elapsed = time.perf_counter() - start_time

    if not history["total"][-1] < history["total"][0]:
        raise RuntimeError(
            "The total loss did not decrease during this run."
        )

    return {
        "history": history,
        "training_seconds": elapsed,
        "quantum_parameter_change_norm": (
            quantum_change_norm(
                model=model,
                initial_quantum_parameters=(
                    initial_quantum_parameters
                ),
            )
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    alpha: float,
    bands: FrequencyBands,
    requested_times: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
    ),
    n_x: int = 201,
    n_t: int = 101,
) -> dict[str, Any]:
    """Evaluate physical and spectral errors on one common grid."""
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

    inputs = torch.cat(
        [x_column, t_column],
        dim=1,
    )

    prediction = model(inputs)
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
    mean_absolute_error = float(
        torch.mean(torch.abs(error)).cpu()
    )
    maximum_absolute_error = float(
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
    x_numpy = x.cpu().numpy()
    t_numpy = t.cpu().numpy()

    spectral_metrics: dict[str, Any] = {}

    for requested_time in requested_times:
        time_index = int(
            np.argmin(
                np.abs(t_numpy - requested_time)
            )
        )
        actual_time = float(
            t_numpy[time_index]
        )

        analysis = analyze_spatial_profile(
            x=x_numpy,
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
                for band_name, band_result
                in analysis[
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
        "spectral_metrics": spectral_metrics,
    }


def run_one_model(
    specification: ModelSpecification,
    seed: int,
    shared_points,
    config: HeatTrainingConfig,
    bands: FrequencyBands,
    output_directory: Path,
) -> FairRunResult:
    """Initialize, train, evaluate, and save one controlled run."""
    # Both models receive deterministic initialization for this seed.
    set_reproducible_seed(seed)
    model = specification.constructor()

    training = train_on_shared_points(
        model=model,
        points=shared_points,
        config=config,
    )

    evaluation = evaluate_model(
        model=model,
        alpha=config.alpha,
        bands=bands,
    )

    model_directory = (
        output_directory
        / f"seed_{seed}"
        / specification.name
    )
    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        model_directory / "model_state.pt",
    )

    with (
        model_directory / "history.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            training["history"],
            file,
            indent=2,
        )

    with (
        model_directory / "spectral_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            evaluation["spectral_metrics"],
            file,
            indent=2,
        )

    final_quantum_gradient = (
        training["history"]["quantum_gradient"][-1]
        if training["history"]["quantum_gradient"]
        else None
    )

    quantum_parameters = (
        quantum_parameter_count(model)
        if isinstance(model, HybridQuantumPINN)
        else 0
    )

    result = FairRunResult(
        model=specification.name,
        architecture=specification.architecture,
        model_family=specification.model_family,
        seed=seed,
        trainable_parameters=parameter_count(model),
        quantum_parameters=quantum_parameters,
        epochs=config.epochs,
        n_collocation=config.n_collocation,
        n_initial=config.n_initial,
        n_boundary=config.n_boundary,
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
        relative_l2=evaluation["relative_l2"],
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

    with (
        model_directory / "run_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
        )

    return result


def aggregate_scalar(
    results: list[FairRunResult],
    field: str,
) -> dict[str, float]:
    """Return mean and sample standard deviation."""
    values = [
        float(getattr(result, field))
        for result in results
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
    results: list[FairRunResult],
    time_key: str,
    band_name: str,
) -> dict[str, float]:
    """Aggregate one stable spectral band error."""
    metric_name = (
        "error_normalized_by_total_reference_energy"
    )

    values = [
        float(
            result
            .spectral_metrics[time_key]
            ["bands"][band_name]
            [metric_name]
        )
        for result in results
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
    results: list[FairRunResult],
) -> dict[str, Any]:
    """Create multi-seed performance and spectral summaries."""
    summary: dict[str, Any] = {}

    for specification in model_specifications():
        selected = [
            result
            for result in results
            if result.model == specification.name
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
            "quantum_parameters": (
                selected[0].quantum_parameters
            ),
            "relative_l2": aggregate_scalar(
                selected,
                "relative_l2",
            ),
            "mean_absolute_error": aggregate_scalar(
                selected,
                "mean_absolute_error",
            ),
            "maximum_absolute_error": aggregate_scalar(
                selected,
                "maximum_absolute_error",
            ),
            "final_total_loss": aggregate_scalar(
                selected,
                "final_total_loss",
            ),
            "final_pde_loss": aggregate_scalar(
                selected,
                "final_pde_loss",
            ),
            "final_initial_loss": aggregate_scalar(
                selected,
                "final_initial_loss",
            ),
            "final_boundary_loss": aggregate_scalar(
                selected,
                "final_boundary_loss",
            ),
            "training_seconds": aggregate_scalar(
                selected,
                "training_seconds",
            ),
            "spectral_errors": {},
        }

        time_keys = sorted(
            selected[0]
            .spectral_metrics
            .keys()
        )

        for time_key in time_keys:
            model_summary[
                "spectral_errors"
            ][time_key] = {
                band_name: (
                    aggregate_spectral_error(
                        results=selected,
                        time_key=time_key,
                        band_name=band_name,
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


def save_csv(
    results: list[FairRunResult],
    output_path: Path,
) -> None:
    """Save scalar run-level metrics as CSV."""
    rows = []

    for result in results:
        row = asdict(result)
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


def save_comparison_plots(
    results: list[FairRunResult],
    summary: dict[str, Any],
    output_directory: Path,
) -> None:
    """Save independent comparison figures."""
    model_names = [
        specification.name
        for specification in model_specifications()
    ]

    seeds = sorted(
        {
            result.seed
            for result in results
        }
    )

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    x_positions = np.arange(len(seeds))
    width = 0.35

    for model_index, model_name in enumerate(
        model_names
    ):
        model_results = {
            result.seed: result
            for result in results
            if result.model == model_name
        }

        offset = (
            model_index
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
        "Fair Heat comparison by random seed"
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

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    means = [
        summary[name]["relative_l2"]["mean"]
        for name in model_names
    ]
    standard_deviations = [
        summary[name]["relative_l2"]["std"]
        for name in model_names
    ]

    axis.bar(
        model_names,
        means,
        yerr=standard_deviations,
        capsize=5,
    )
    axis.set_ylabel("Relative L2 error")
    axis.set_title(
        "Mean relative L2 error across seeds"
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

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    time_key = "0.500000"
    band_names = [
        "low",
        "middle",
        "high",
    ]
    x_positions = np.arange(
        len(band_names)
    )

    for model_index, model_name in enumerate(
        model_names
    ):
        offset = (
            model_index
            - (len(model_names) - 1) / 2
        ) * width

        spectral_summary = summary[
            model_name
        ]["spectral_errors"][time_key]

        axis.bar(
            x_positions + offset,
            [
                spectral_summary[band]["mean"]
                for band in band_names
            ],
            width=width,
            yerr=[
                spectral_summary[band]["std"]
                for band in band_names
            ],
            capsize=4,
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
        "Spectral errors at t=0.5"
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


def write_readme(
    summary: dict[str, Any],
    config: HeatTrainingConfig,
    seeds: list[int],
    output_path: Path,
) -> None:
    """Write an honest experiment report."""
    classical = summary[
        "classical_bottleneck"
    ]
    quantum = summary[
        "four_qubit_qapinn"
    ]

    lines = [
        "# Fair classical bottleneck vs four-qubit QAPINN pilot",
        "",
        "## Controlled conditions",
        "",
        f"- Seeds: {seeds}",
        f"- Epochs: {config.epochs}",
        f"- Collocation points: {config.n_collocation}",
        f"- Initial-condition points: {config.n_initial}",
        f"- Boundary points: {config.n_boundary}",
        f"- Learning rate: {config.learning_rate}",
        (
            "- Each model used exactly the same sampled training "
            "coordinates within each seed."
        ),
        "",
        "## Aggregate relative L2 error",
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
        "## Parameter counts",
        "",
        (
            "- Classical bottleneck: "
            f"{classical['trainable_parameters']}"
        ),
        (
            "- Four-qubit QAPINN: "
            f"{quantum['trainable_parameters']} total, "
            f"{quantum['quantum_parameters']} quantum-circuit parameters"
        ),
        "",
        "## Interpretation constraint",
        "",
        (
            "This is a small matched pilot, not evidence of quantum "
            "advantage. The models are dimension-matched at the "
            "four-value interface but are not exactly parameter-matched "
            "(109 vs 125 trainable parameters)."
        ),
        (
            "The quantum backend is an ideal differentiable state-vector "
            "simulator with no finite shots, device noise, or hardware."
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
            "four-qubit QAPINN Heat pilot."
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
        default=Path("outputs/fair_classical_quantum_heat"),
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

    selected_bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: list[FairRunResult] = []

    for seed in arguments.seeds:
        # Sample once, then give the exact same tensors to both models.
        set_reproducible_seed(seed)

        shared_points = sample_heat_training_points(
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        for specification in model_specifications():
            print(
                "\n========================================"
            )
            print(
                f"Seed {seed} | {specification.name}"
            )
            print(
                "========================================"
            )

            result = run_one_model(
                specification=specification,
                seed=seed,
                shared_points=shared_points,
                config=config,
                bands=selected_bands,
                output_directory=(
                    arguments.output_directory
                ),
            )

            all_results.append(result)

            print(
                f"Completed {specification.name}, "
                f"seed {seed}: "
                f"relative L2={result.relative_l2:.6f}, "
                f"time={result.training_seconds:.2f}s"
            )

    summary = summarize_results(
        all_results
    )

    with (
        arguments.output_directory
        / "runs.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            [
                asdict(result)
                for result in all_results
            ],
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
                "frequency_bands": (
                    asdict(selected_bands)
                ),
                "shared_points_within_seed": True,
            },
            file,
            indent=2,
        )

    save_csv(
        results=all_results,
        output_path=(
            arguments.output_directory
            / "runs.csv"
        ),
    )

    save_comparison_plots(
        results=all_results,
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

    print(
        "\nAggregate comparison"
    )

    for model_name, values in summary.items():
        print(
            f"{model_name:24s} | "
            f"params={values['trainable_parameters']:3d} | "
            f"relative L2="
            f"{values['relative_l2']['mean']:.6f} ± "
            f"{values['relative_l2']['std']:.6f} | "
            f"training time="
            f"{values['training_seconds']['mean']:.2f}s"
        )

    print(
        "\nSaved outputs to: "
        f"{arguments.output_directory}"
    )
