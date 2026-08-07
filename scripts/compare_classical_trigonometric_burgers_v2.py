from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from statistics import mean, stdev
from typing import Any, Callable
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from classical_trigonometric import (
    ClassicalTrigonometricPINN,
    count_fixed_phase_values,
)
from quantum_layer import HybridQuantumPINN
from spectra import FrequencyBands
from train_burgers import (
    BurgersRunResult,
    BurgersTrainingConfig,
    load_reference,
    run_one_model,
    sample_burgers_training_points,
)
from train_heat import set_reproducible_seed


TRIG_OUTPUT = Path(
    "outputs/classical_trig_burgers_runs_v2"
)
PRIOR_ABLATION = Path(
    "outputs/restored_improved_burgers_quantum_ablation"
)
FINAL_OUTPUT = Path(
    "outputs/burgers_classical_trig_control_v2"
)


@dataclass(frozen=True)
class TrigSpecification:
    """One fixed classical trigonometric control."""

    name: str
    architecture: str
    constructor: Callable[[], nn.Module]


def trig_specifications() -> list[TrigSpecification]:
    """Return local and ring-coupled classical trigonometric controls."""
    return [
        TrigSpecification(
            name="classical_trig_local",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "fixed local cosine products -> 4 -> 8 -> 1"
            ),
            constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="local",
            ),
        ),
        TrigSpecification(
            name="classical_trig_ring",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "fixed ring cosine products -> 4 -> 8 -> 1"
            ),
            constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="ring",
            ),
        ),
    ]


def verify_phase_initialization_match() -> float:
    """Verify that fixed trig phases use the quantum circuit's RNG position.

    Because both models construct the same surrounding linear layers before
    creating a (2,4,2) middle tensor and use Uniform(-0.1,0.1), their fixed
    phase values should equal the frozen circuit's initial quantum angles for
    the same seed.
    """
    seed = 2026

    set_reproducible_seed(seed)
    quantum_model = HybridQuantumPINN(
        in_features=2,
        classical_width=8,
        n_qubits=4,
        quantum_depth=2,
        entanglement="ring",
        out_features=1,
    )

    set_reproducible_seed(seed)
    trig_model = ClassicalTrigonometricPINN(
        connectivity="ring",
    )

    difference = torch.max(
        torch.abs(
            quantum_model.quantum_layer.quantum_weights.detach()
            - trig_model.feature_map.fixed_phases.detach()
        )
    ).item()

    if difference > 0.0:
        raise RuntimeError(
            "Fixed classical phases do not match the frozen quantum "
            f"initialization; maximum difference={difference}."
        )

    return difference


def aggregate(values: list[float]) -> dict[str, float]:
    """Return mean and sample standard deviation."""
    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def standardize_prior_runs() -> list[dict[str, Any]]:
    """Load the four existing improved Burgers ablation variants."""
    path = PRIOR_ABLATION / "runs.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Prior ablation runs are missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def run_trigonometric_controls() -> list[BurgersRunResult]:
    """Train both classical trigonometric controls for all three seeds."""
    if TRIG_OUTPUT.exists():
        shutil.rmtree(TRIG_OUTPUT)

    TRIG_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_reference(
        Path("data/burgers_reference.npz")
    )

    config = BurgersTrainingConfig(
        viscosity=reference.viscosity,
        epochs=500,
        learning_rate=1.0e-3,
        n_collocation=128,
        n_initial=32,
        n_boundary=32,
        pde_weight=1.0,
        initial_weight=10.0,
        boundary_weight=1.0,
        log_every=500,
    )

    bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    runs: list[BurgersRunResult] = []

    for seed in (11, 22, 33):
        # Recreate the identical coordinate samples used by all prior models.
        set_reproducible_seed(seed)
        shared_points = sample_burgers_training_points(
            config=config,
            dtype=torch.float32,
        )

        for specification in trig_specifications():
            print(
                "\n=============================================="
            )
            print(
                f"Seed {seed} | {specification.name}"
            )
            print(
                "=============================================="
            )

            result = run_one_model(
                specification=specification,
                seed=seed,
                points=shared_points,
                config=config,
                reference=reference,
                bands=bands,
                output_directory=TRIG_OUTPUT,
            )
            runs.append(result)

            print(
                f"Completed {specification.name}: "
                f"relative L2={result.relative_l2:.6f}, "
                f"time={result.training_seconds:.2f}s"
            )

    with (
        TRIG_OUTPUT / "runs.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                asdict(run)
                for run in runs
            ],
            file,
            indent=2,
        )

    return runs


def summarize_model(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate scalar and spectral metrics for one model."""
    scalar_fields = [
        "final_total_loss",
        "final_pde_loss",
        "final_initial_loss",
        "final_boundary_loss",
        "relative_l2",
        "mean_absolute_error",
        "maximum_absolute_error",
        "training_seconds",
    ]

    summary: dict[str, Any] = {
        "number_of_seeds": len(selected),
        "seeds": [
            int(run["seed"])
            for run in selected
        ],
        "architecture": selected[0]["architecture"],
        "trainable_parameters": int(
            selected[0]["trainable_parameters"]
        ),
        "quantum_parameters": int(
            selected[0]["quantum_parameters"]
        ),
    }

    for field in scalar_fields:
        summary[field] = aggregate(
            [
                float(run[field])
                for run in selected
            ]
        )

    summary["spectral_errors"] = {}

    for time_key in sorted(
        selected[0]["spectral_metrics"].keys()
    ):
        summary["spectral_errors"][time_key] = {}

        for band in (
            "low",
            "middle",
            "high",
        ):
            metric = (
                "error_normalized_by_"
                "total_reference_energy"
            )
            summary[
                "spectral_errors"
            ][time_key][band] = aggregate(
                [
                    float(
                        run[
                            "spectral_metrics"
                        ][time_key][
                            "bands"
                        ][band][metric]
                    )
                    for run in selected
                ]
            )

    return summary


def paired_comparison(
    runs: list[dict[str, Any]],
    first_model: str,
    second_model: str,
) -> dict[str, Any]:
    """Compare model errors within each identical seed."""
    lookup = {
        (
            run["model"],
            int(run["seed"]),
        ): run
        for run in runs
    }

    rows = []
    differences = []

    for seed in (11, 22, 33):
        first = float(
            lookup[
                (first_model, seed)
            ]["relative_l2"]
        )
        second = float(
            lookup[
                (second_model, seed)
            ]["relative_l2"]
        )
        difference = first - second
        differences.append(difference)

        rows.append(
            {
                "seed": seed,
                "first_relative_l2": first,
                "second_relative_l2": second,
                "difference_first_minus_second": difference,
            }
        )

    return {
        "first_model": first_model,
        "second_model": second_model,
        "rows": rows,
        "difference": aggregate(differences),
        "first_model_wins": sum(
            value < 0.0
            for value in differences
        ),
        "second_model_wins": sum(
            value > 0.0
            for value in differences
        ),
    }


def consolidate(
    trig_runs: list[BurgersRunResult],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Merge classical trigonometric controls with prior ablations."""
    prior_runs = standardize_prior_runs()

    all_runs = prior_runs + [
        asdict(run)
        for run in trig_runs
    ]

    model_order = [
        "classical_bottleneck",
        "classical_trig_local",
        "classical_trig_ring",
        "qapinn_no_entanglement",
        "qapinn_ring_trainable",
        "qapinn_ring_frozen",
    ]

    expected = {
        (model, seed)
        for model in model_order
        for seed in (11, 22, 33)
    }
    observed = {
        (
            run["model"],
            int(run["seed"]),
        )
        for run in all_runs
    }

    if expected != observed:
        raise RuntimeError(
            "Merged model/seed set is incomplete. "
            f"Missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )

    model_summary = {
        model: summarize_model(
            [
                run
                for run in all_runs
                if run["model"] == model
            ]
        )
        for model in model_order
    }

    paired = {
        "classical_trig_ring_minus_frozen_quantum_ring": (
            paired_comparison(
                all_runs,
                "classical_trig_ring",
                "qapinn_ring_frozen",
            )
        ),
        "classical_trig_local_minus_quantum_no_entanglement": (
            paired_comparison(
                all_runs,
                "classical_trig_local",
                "qapinn_no_entanglement",
            )
        ),
        "classical_trig_ring_minus_classical_bottleneck": (
            paired_comparison(
                all_runs,
                "classical_trig_ring",
                "classical_bottleneck",
            )
        ),
        "frozen_quantum_ring_minus_classical_bottleneck": (
            paired_comparison(
                all_runs,
                "qapinn_ring_frozen",
                "classical_bottleneck",
            )
        ),
    }

    summary = {
        "experiment": {
            "pde": (
                "u_t + u*u_x = viscosity*u_xx"
            ),
            "epochs": 500,
            "initial_condition_weight": 10.0,
            "seeds": [11, 22, 33],
            "shared_points_within_seed": True,
            "fixed_phase_count": 16,
            "phase_initialization_match_verified": True,
            "classical_trig_description": (
                "Explicit cosine-product maps; no qubits, "
                "state vectors, gates, or quantum measurements."
            ),
        },
        "model_order": model_order,
        "models": model_summary,
        "paired_comparisons": paired,
    }

    return all_runs, summary


def copy_artifacts() -> None:
    """Copy trained states and predictions into one final package."""
    if FINAL_OUTPUT.exists():
        shutil.rmtree(
            FINAL_OUTPUT
        )

    FINAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in (11, 22, 33):
        destination = (
            FINAL_OUTPUT
            / f"seed_{seed}"
        )
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        prior_seed = (
            PRIOR_ABLATION
            / f"seed_{seed}"
        )

        for model_name in (
            "classical_bottleneck",
            "qapinn_no_entanglement",
            "qapinn_ring_trainable",
            "qapinn_ring_frozen",
        ):
            shutil.copytree(
                prior_seed / model_name,
                destination / model_name,
            )

        trig_seed = (
            TRIG_OUTPUT
            / f"seed_{seed}"
        )

        for model_name in (
            "classical_trig_local",
            "classical_trig_ring",
        ):
            shutil.copytree(
                trig_seed / model_name,
                destination / model_name,
            )


def save_runs_csv(
    runs: list[dict[str, Any]],
) -> None:
    """Save scalar run metrics."""
    rows = []

    for run in runs:
        row = {
            key: value
            for key, value in run.items()
            if key != "spectral_metrics"
        }
        rows.append(row)

    with (
        FINAL_OUTPUT
        / "runs.csv"
    ).open(
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


def save_error_plots(
    runs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Save seed-level, aggregate, and spectral figures."""
    model_order = summary[
        "model_order"
    ]
    display = {
        "classical_bottleneck": "Classical bottleneck",
        "classical_trig_local": "Classical trig: local",
        "classical_trig_ring": "Classical trig: ring",
        "qapinn_no_entanglement": "Quantum: no entanglement",
        "qapinn_ring_trainable": "Quantum: trainable ring",
        "qapinn_ring_frozen": "Quantum: frozen ring",
    }

    seeds = [11, 22, 33]
    positions = np.arange(
        len(seeds)
    )
    width = 0.13

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    for index, model in enumerate(
        model_order
    ):
        lookup = {
            int(run["seed"]): run
            for run in runs
            if run["model"] == model
        }
        offset = (
            index
            - (
                len(model_order) - 1
            ) / 2
        ) * width

        axis.bar(
            positions + offset,
            [
                lookup[seed][
                    "relative_l2"
                ]
                for seed in seeds
            ],
            width=width,
            label=display[model],
        )

    axis.set_xticks(
        positions,
        [
            str(seed)
            for seed in seeds
        ],
    )
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Relative L2 error")
    axis.set_title(
        "Burgers: quantum and classical trigonometric controls"
    )
    axis.legend(
        fontsize=7,
        ncol=2,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        FINAL_OUTPUT
        / "relative_l2_by_seed.png",
        dpi=200,
    )
    plt.close(figure)

    models = summary["models"]

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        [
            display[model]
            for model in model_order
        ],
        [
            models[model][
                "relative_l2"
            ]["mean"]
            for model in model_order
        ],
        yerr=[
            models[model][
                "relative_l2"
            ]["std"]
            for model in model_order
        ],
        capsize=4,
    )
    axis.set_ylabel(
        "Relative L2 error"
    )
    axis.set_title(
        "Mean Burgers error across seeds"
    )
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        FINAL_OUTPUT
        / "relative_l2_mean_std.png",
        dpi=200,
    )
    plt.close(figure)

    bands = [
        "low",
        "middle",
        "high",
    ]
    positions = np.arange(
        len(bands)
    )

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    for index, model in enumerate(
        model_order
    ):
        offset = (
            index
            - (
                len(model_order) - 1
            ) / 2
        ) * width
        values = models[
            model
        ]["spectral_errors"][
            "0.500000"
        ]

        axis.bar(
            positions + offset,
            [
                values[band]["mean"]
                for band in bands
            ],
            width=width,
            yerr=[
                values[band]["std"]
                for band in bands
            ],
            capsize=2,
            label=display[model],
        )

    axis.set_xticks(
        positions,
        bands,
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
    axis.legend(
        fontsize=7,
        ncol=2,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        FINAL_OUTPUT
        / "spectral_errors_t_0p5.png",
        dpi=200,
    )
    plt.close(figure)

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        [
            display[model]
            for model in model_order
        ],
        [
            models[model][
                "training_seconds"
            ]["mean"]
            for model in model_order
        ],
        yerr=[
            models[model][
                "training_seconds"
            ]["std"]
            for model in model_order
        ],
        capsize=4,
    )
    axis.set_ylabel(
        "Training time (seconds)"
    )
    axis.set_title(
        "Computational cost of quantum and classical controls"
    )
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        FINAL_OUTPUT
        / "training_time_mean_std.png",
        dpi=200,
    )
    plt.close(figure)


def write_readme(
    summary: dict[str, Any],
) -> None:
    """Write a neutral report; final interpretation is added after execution."""
    models = summary["models"]

    lines = [
        "# Classical trigonometric control for the Burgers QAPINN",
        "",
        "## Purpose",
        "",
        (
            "Test whether the frozen ring QAPINN's improvement can be "
            "reproduced by an explicit classical cosine-product feature map."
        ),
        "",
        "## Fairness controls",
        "",
        "- Same 109 trainable parameters for the classical bottleneck, "
        "classical trigonometric controls, and frozen ring QAPINN.",
        "- Sixteen fixed phase constants in both trigonometric controls.",
        "- The fixed phase tensor exactly matches the frozen quantum circuit's "
        "initial angle tensor for the same random seed.",
        "- Same 500 epochs, IC weight 10, coordinates, optimizer, and seeds.",
        "",
        "## Mean relative L2",
        "",
    ]

    for model in summary["model_order"]:
        values = models[
            model
        ]["relative_l2"]

        lines.append(
            f"- {model}: "
            f"{values['mean']:.6f} ± "
            f"{values['std']:.6f}"
        )

    (
        FINAL_OUTPUT
        / "README.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    phase_difference = (
        verify_phase_initialization_match()
    )

    print(
        "Maximum fixed-phase difference between "
        "matched quantum and classical models: "
        f"{phase_difference:.3e}"
    )

    trig_runs = (
        run_trigonometric_controls()
    )
    all_runs, summary = consolidate(
        trig_runs
    )

    copy_artifacts()

    with (
        FINAL_OUTPUT
        / "runs.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_runs,
            file,
            indent=2,
        )

    with (
        FINAL_OUTPUT
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

    save_runs_csv(all_runs)
    save_error_plots(
        all_runs,
        summary,
    )
    write_readme(summary)

    zip_path = Path(
        "outputs/burgers_classical_trig_control_v2.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in sorted(
            FINAL_OUTPUT.rglob("*")
        ):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(
                        FINAL_OUTPUT
                    ),
                )

    print(
        "\nConsolidated Burgers classical-trigonometric control"
    )

    for model in summary[
        "model_order"
    ]:
        values = summary[
            "models"
        ][model]

        print(
            f"{model:30s} | "
            f"trainable="
            f"{values['trainable_parameters']:3d} | "
            f"relative L2="
            f"{values['relative_l2']['mean']:.6f} ± "
            f"{values['relative_l2']['std']:.6f} | "
            f"time="
            f"{values['training_seconds']['mean']:.2f}s"
        )

    print(
        "\nCreated archive: "
        f"{zip_path}"
    )
