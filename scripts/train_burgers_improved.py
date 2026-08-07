from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from spectra import FrequencyBands
from train_heat import set_reproducible_seed
from train_burgers import (
    BurgersRunResult,
    BurgersTrainingConfig,
    load_reference,
    model_specifications,
    run_one_model,
    sample_burgers_training_points,
    save_figures,
    save_scalar_csv,
    summarize_results,
)


def compare_with_baseline(
    improved_summary: dict[str, Any],
    baseline_summary_path: Path,
) -> dict[str, Any] | None:
    """Compare improved results with the earlier 100-epoch, unit-weight pilot."""
    if not baseline_summary_path.exists():
        return None

    with baseline_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline_summary = json.load(file)

    comparison: dict[str, Any] = {}

    for model_name, improved_values in improved_summary.items():
        baseline_values = baseline_summary[model_name]

        baseline_error = float(
            baseline_values["relative_l2"]["mean"]
        )
        improved_error = float(
            improved_values["relative_l2"]["mean"]
        )

        baseline_ic = float(
            baseline_values["final_initial_loss"]["mean"]
        )
        improved_ic = float(
            improved_values["final_initial_loss"]["mean"]
        )

        comparison[model_name] = {
            "baseline_relative_l2_mean": baseline_error,
            "improved_relative_l2_mean": improved_error,
            "absolute_relative_l2_change": (
                improved_error - baseline_error
            ),
            "relative_l2_reduction_fraction": (
                (baseline_error - improved_error)
                / baseline_error
            ),
            "baseline_final_initial_loss_mean": baseline_ic,
            "improved_final_initial_loss_mean": improved_ic,
            "initial_loss_reduction_fraction": (
                (baseline_ic - improved_ic)
                / baseline_ic
            ),
        }

    return comparison


def save_baseline_comparison_csv(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Save baseline-versus-improved summary as CSV."""
    rows = [
        {
            "model": model_name,
            **values,
        }
        for model_name, values in comparison.items()
    ]

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


def save_baseline_comparison_figure(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot baseline and improved mean relative L2 errors."""
    model_names = list(comparison.keys())
    positions = np.arange(len(model_names))
    width = 0.36

    baseline = [
        comparison[name][
            "baseline_relative_l2_mean"
        ]
        for name in model_names
    ]
    improved = [
        comparison[name][
            "improved_relative_l2_mean"
        ]
        for name in model_names
    ]

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    axis.bar(
        positions - width / 2,
        baseline,
        width=width,
        label="100 epochs, IC weight 1",
    )
    axis.bar(
        positions + width / 2,
        improved,
        width=width,
        label="500 epochs, IC weight 10",
    )

    axis.set_xticks(
        positions,
        model_names,
    )
    axis.set_ylabel("Mean relative L2 error")
    axis.set_title(
        "Effect of longer training and stronger initial-condition loss"
    )
    axis.legend()
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)


def write_improved_readme(
    summary: dict[str, Any],
    comparison: dict[str, Any] | None,
    config: BurgersTrainingConfig,
    seeds: list[int],
    output_path: Path,
) -> None:
    """Write an honest report for the improved training experiment."""
    lines = [
        "# Improved Burgers PINN training",
        "",
        "## Changed settings",
        "",
        f"- Epochs: {config.epochs}",
        f"- Initial-condition loss weight: {config.initial_weight}",
        f"- PDE loss weight: {config.pde_weight}",
        f"- Boundary loss weight: {config.boundary_weight}",
        "",
        "## Unchanged controls",
        "",
        f"- Seeds: {seeds}",
        f"- Collocation points: {config.n_collocation}",
        f"- Initial-condition points: {config.n_initial}",
        f"- Boundary points: {config.n_boundary}",
        "- Identical sampled coordinates for both models within each seed.",
        "- Same optimizer, learning rate, architectures, and reference solution.",
        "",
        "## Aggregate results",
        "",
    ]

    for model_name, values in summary.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                (
                    "- Relative L2: "
                    f"{values['relative_l2']['mean']:.6f} ± "
                    f"{values['relative_l2']['std']:.6f}"
                ),
                (
                    "- Final unweighted initial-condition loss: "
                    f"{values['final_initial_loss']['mean']:.6f} ± "
                    f"{values['final_initial_loss']['std']:.6f}"
                ),
                (
                    "- Final PDE loss: "
                    f"{values['final_pde_loss']['mean']:.6f} ± "
                    f"{values['final_pde_loss']['std']:.6f}"
                ),
                "",
            ]
        )

    if comparison is not None:
        lines.extend(
            [
                "## Comparison with the earlier pilot",
                "",
            ]
        )

        for model_name, values in comparison.items():
            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    (
                        "- Relative L2 reduction: "
                        f"{100.0 * values['relative_l2_reduction_fraction']:.2f}%"
                    ),
                    (
                        "- Initial-condition loss reduction: "
                        f"{100.0 * values['initial_loss_reduction_fraction']:.2f}%"
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation constraint",
            "",
            (
                "Because epochs and initial-condition weighting changed "
                "together, this experiment measures their combined effect. "
                "It does not isolate which change caused the improvement."
            ),
            (
                "The quantum backend remains an ideal differentiable "
                "state-vector simulator rather than quantum hardware."
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
            "Train matched Burgers classical and quantum PINNs "
            "for 500 epochs with initial-condition weight 10."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/burgers_reference.npz"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path(
            "outputs/fair_classical_quantum_burgers/summary.json"
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
        default=500,
    )
    parser.add_argument(
        "--initial-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/improved_burgers_training"),
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
        pde_weight=1.0,
        initial_weight=arguments.initial_weight,
        boundary_weight=1.0,
        log_every=100,
    )

    bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs: list[BurgersRunResult] = []

    for seed in arguments.seeds:
        set_reproducible_seed(seed)

        shared_points = sample_burgers_training_points(
            config=config,
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

            runs.append(run)

            print(
                f"Completed {specification.name}, seed {seed}: "
                f"relative L2={run.relative_l2:.6f}, "
                f"time={run.training_seconds:.2f}s"
            )

    summary = summarize_results(runs)

    comparison = compare_with_baseline(
        improved_summary=summary,
        baseline_summary_path=(
            arguments.baseline_summary
        ),
    )

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
                "reference_path": str(
                    arguments.reference
                ),
                "shared_points_within_seed": True,
            },
            file,
            indent=2,
        )

    save_scalar_csv(
        runs=runs,
        output_path=(
            arguments.output_directory
            / "runs.csv"
        ),
    )

    save_figures(
        runs=runs,
        summary=summary,
        reference=reference,
        output_directory=(
            arguments.output_directory
        ),
    )

    if comparison is not None:
        with (
            arguments.output_directory
            / "baseline_comparison.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                comparison,
                file,
                indent=2,
            )

        save_baseline_comparison_csv(
            comparison=comparison,
            output_path=(
                arguments.output_directory
                / "baseline_comparison.csv"
            ),
        )

        save_baseline_comparison_figure(
            comparison=comparison,
            output_path=(
                arguments.output_directory
                / "baseline_vs_improved_relative_l2.png"
            ),
        )

    write_improved_readme(
        summary=summary,
        comparison=comparison,
        config=config,
        seeds=arguments.seeds,
        output_path=(
            arguments.output_directory
            / "README.md"
        ),
    )

    print("\nImproved Burgers aggregate results")

    for model_name, values in summary.items():
        print(
            f"{model_name:24s} | "
            f"relative L2="
            f"{values['relative_l2']['mean']:.6f} ± "
            f"{values['relative_l2']['std']:.6f} | "
            f"IC loss="
            f"{values['final_initial_loss']['mean']:.6f} | "
            f"time="
            f"{values['training_seconds']['mean']:.2f}s"
        )

    if comparison is not None:
        print("\nChange from earlier 100-epoch pilot")

        for model_name, values in comparison.items():
            print(
                f"{model_name:24s} | "
                f"L2 reduction="
                f"{100.0 * values['relative_l2_reduction_fraction']:.2f}% | "
                f"IC reduction="
                f"{100.0 * values['initial_loss_reduction_fraction']:.2f}%"
            )

    print(
        "\nSaved outputs to: "
        f"{arguments.output_directory}"
    )
