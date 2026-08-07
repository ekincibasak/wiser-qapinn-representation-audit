from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

import torch
from torch import nn

from classical_bottleneck import (
    ClassicalBottleneckPINN,
    count_trainable_parameters,
)
from classical_mlp import ClassicalMLP
from train_heat import (
    HeatTrainingConfig,
    evaluate_heat_pinn,
    parameter_change_norm,
    set_reproducible_seed,
    train_heat_pinn,
)


@dataclass(frozen=True)
class ModelSpecification:
    """Definition of one classical control model."""

    name: str
    architecture: str
    constructor: Callable[[], nn.Module]


@dataclass(frozen=True)
class RunResult:
    """Metrics from one model and one random seed."""

    model: str
    architecture: str
    seed: int
    trainable_parameters: int

    initial_total_loss: float
    final_total_loss: float
    final_pde_loss: float
    final_initial_loss: float
    final_boundary_loss: float
    parameter_change_norm: float

    relative_l2: float
    mean_absolute_error: float
    maximum_absolute_error: float


def model_specifications() -> list[ModelSpecification]:
    """Return the three classical models used in the comparison."""
    return [
        ModelSpecification(
            name="large_mlp",
            architecture="2 -> 16 -> 16 -> 1",
            constructor=lambda: ClassicalMLP(
                in_features=2,
                out_features=1,
                hidden_layers=(16, 16),
            ),
        ),
        ModelSpecification(
            name="parameter_matched_mlp",
            architecture="2 -> 8 -> 8 -> 1",
            constructor=lambda: ClassicalMLP(
                in_features=2,
                out_features=1,
                hidden_layers=(8, 8),
            ),
        ),
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
    ]


def run_one_experiment(
    specification: ModelSpecification,
    seed: int,
    base_config: HeatTrainingConfig,
) -> RunResult:
    """Train and evaluate one model with one random seed."""
    config = HeatTrainingConfig(
        alpha=base_config.alpha,
        epochs=base_config.epochs,
        learning_rate=base_config.learning_rate,
        n_collocation=base_config.n_collocation,
        n_initial=base_config.n_initial,
        n_boundary=base_config.n_boundary,
        x_min=base_config.x_min,
        x_max=base_config.x_max,
        t_min=base_config.t_min,
        t_max=base_config.t_max,
        pde_weight=base_config.pde_weight,
        initial_weight=base_config.initial_weight,
        boundary_weight=base_config.boundary_weight,
        seed=seed,
        log_every=base_config.log_every,
    )

    # Reproducible model initialization.
    set_reproducible_seed(seed)
    model = specification.constructor()

    initial_parameters = [
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    ]

    # train_heat_pinn resets the same seed before point sampling.
    # Therefore, all three models use identical sampled coordinates
    # for a given seed.
    history = train_heat_pinn(
        model=model,
        config=config,
        device="cpu",
    )

    metrics = evaluate_heat_pinn(
        model=model,
        alpha=config.alpha,
        device="cpu",
        n_x=201,
        n_t=101,
    )

    change_norm = parameter_change_norm(
        initial_parameters=initial_parameters,
        model=model,
    )

    if not history.total_loss[-1] < history.total_loss[0]:
        raise RuntimeError(
            f"{specification.name}, seed {seed}: "
            "total loss did not decrease."
        )

    if not math.isfinite(metrics.relative_l2):
        raise RuntimeError(
            f"{specification.name}, seed {seed}: "
            "relative L2 is not finite."
        )

    return RunResult(
        model=specification.name,
        architecture=specification.architecture,
        seed=seed,
        trainable_parameters=count_trainable_parameters(model),
        initial_total_loss=history.total_loss[0],
        final_total_loss=history.total_loss[-1],
        final_pde_loss=history.pde_loss[-1],
        final_initial_loss=history.initial_loss[-1],
        final_boundary_loss=history.boundary_loss[-1],
        parameter_change_norm=change_norm,
        relative_l2=metrics.relative_l2,
        mean_absolute_error=metrics.mean_absolute_error,
        maximum_absolute_error=metrics.maximum_absolute_error,
    )


def summarize_results(
    results: list[RunResult],
) -> dict[str, dict[str, float | int | str]]:
    """Calculate mean and sample standard deviation for each model."""
    summary: dict[str, dict[str, float | int | str]] = {}

    for specification in model_specifications():
        model_results = [
            result
            for result in results
            if result.model == specification.name
        ]

        if not model_results:
            raise ValueError(
                f"No results found for {specification.name}."
            )

        def aggregate(field: str) -> tuple[float, float]:
            values = [
                float(getattr(result, field))
                for result in model_results
            ]
            standard_deviation = (
                stdev(values) if len(values) > 1 else 0.0
            )
            return mean(values), standard_deviation

        relative_l2_mean, relative_l2_std = aggregate(
            "relative_l2"
        )
        mae_mean, mae_std = aggregate(
            "mean_absolute_error"
        )
        max_error_mean, max_error_std = aggregate(
            "maximum_absolute_error"
        )
        total_loss_mean, total_loss_std = aggregate(
            "final_total_loss"
        )
        pde_mean, pde_std = aggregate(
            "final_pde_loss"
        )
        initial_mean, initial_std = aggregate(
            "final_initial_loss"
        )
        boundary_mean, boundary_std = aggregate(
            "final_boundary_loss"
        )

        summary[specification.name] = {
            "architecture": specification.architecture,
            "trainable_parameters": (
                model_results[0].trainable_parameters
            ),
            "number_of_seeds": len(model_results),
            "relative_l2_mean": relative_l2_mean,
            "relative_l2_std": relative_l2_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "maximum_error_mean": max_error_mean,
            "maximum_error_std": max_error_std,
            "final_total_loss_mean": total_loss_mean,
            "final_total_loss_std": total_loss_std,
            "final_pde_loss_mean": pde_mean,
            "final_pde_loss_std": pde_std,
            "final_initial_loss_mean": initial_mean,
            "final_initial_loss_std": initial_std,
            "final_boundary_loss_mean": boundary_mean,
            "final_boundary_loss_std": boundary_std,
        }

    return summary


def save_results(
    results: list[RunResult],
    summary: dict[str, dict[str, float | int | str]],
    config: HeatTrainingConfig,
    output_directory: Path,
) -> None:
    """Save raw runs, aggregate statistics, and experiment settings."""
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "runs.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
        )

    with (output_directory / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    with (output_directory / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(config), file, indent=2)

    fieldnames = list(asdict(results[0]).keys())

    with (output_directory / "runs.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "# Classical Heat PINN comparison",
        "",
        (
            "All models used identical sampled training coordinates "
            "within each seed."
        ),
        "",
        "| Model | Parameters | Relative L2 mean ± SD | Final loss mean ± SD |",
        "|---|---:|---:|---:|",
    ]

    for specification in model_specifications():
        model_summary = summary[specification.name]
        lines.append(
            "| "
            f"{specification.name} | "
            f"{model_summary['trainable_parameters']} | "
            f"{model_summary['relative_l2_mean']:.6f} ± "
            f"{model_summary['relative_l2_std']:.6f} | "
            f"{model_summary['final_total_loss_mean']:.6f} ± "
            f"{model_summary['final_total_loss_std']:.6f} |"
        )

    lines.extend(
        [
            "",
            (
                "This is a preliminary multi-seed comparison. "
                "It does not establish statistical significance or "
                "quantum advantage."
            ),
        ]
    )

    (output_directory / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare large, parameter-matched, and bottleneck "
            "classical Heat PINNs."
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
        default=500,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/classical_heat_comparison"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    base_config = HeatTrainingConfig(
        epochs=arguments.epochs,
        learning_rate=1.0e-3,
        n_collocation=512,
        n_initial=128,
        n_boundary=128,
        seed=arguments.seeds[0],
        log_every=arguments.epochs,
    )

    all_results: list[RunResult] = []

    for seed in arguments.seeds:
        for specification in model_specifications():
            print(
                "\nRunning "
                f"{specification.name} with seed {seed}"
            )
            result = run_one_experiment(
                specification=specification,
                seed=seed,
                base_config=base_config,
            )
            all_results.append(result)
            print(
                f"Completed {specification.name}, seed {seed}: "
                f"relative L2={result.relative_l2:.6f}"
            )

    aggregate_summary = summarize_results(all_results)

    save_results(
        results=all_results,
        summary=aggregate_summary,
        config=base_config,
        output_directory=arguments.output_directory,
    )

    print("\nAggregate results")
    for model_name, values in aggregate_summary.items():
        print(
            f"{model_name:24s} | "
            f"parameters={values['trainable_parameters']:3d} | "
            f"relative L2="
            f"{values['relative_l2_mean']:.6f} ± "
            f"{values['relative_l2_std']:.6f}"
        )

    print(
        "\nSaved comparison to: "
        f"{arguments.output_directory}"
    )
