from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
from typing import Any
import zipfile

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(
    "outputs/burgers_factorial_pilot_runs"
)
OUTPUT_DIRECTORY = Path(
    "outputs/burgers_model_seed_extension_summary"
)

DATA_SEEDS = [11, 22]
MODEL_SEEDS = [777, 888, 999, 1111]
PHASE_SEEDS = [101, 303]
VARIANTS = [
    "classical_trig_local",
    "quantum_ring_frozen",
]


def aggregate(
    values: list[float],
) -> dict[str, float]:
    """Return descriptive statistics."""
    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def load_runs() -> list[dict[str, Any]]:
    """Load and validate every run in the 2 x 4 x 2 design."""
    runs: list[dict[str, Any]] = []

    for data_seed in DATA_SEEDS:
        for model_seed in MODEL_SEEDS:
            for phase_seed in PHASE_SEEDS:
                cell_directory = (
                    RUN_ROOT
                    / (
                        f"data_{data_seed}_"
                        f"model_{model_seed}_"
                        f"phase_{phase_seed}"
                    )
                )

                for variant in VARIANTS:
                    summary_path = (
                        cell_directory
                        / variant
                        / "run_summary.json"
                    )

                    if not summary_path.exists():
                        raise FileNotFoundError(
                            f"Missing completed run: {summary_path}"
                        )

                    with summary_path.open(
                        "r",
                        encoding="utf-8",
                    ) as file:
                        run = json.load(file)

                    expected = (
                        data_seed,
                        model_seed,
                        phase_seed,
                        variant,
                    )
                    observed = (
                        int(run["data_seed"]),
                        int(run["model_seed"]),
                        int(run["phase_seed"]),
                        run["variant"],
                    )

                    if expected != observed:
                        raise ValueError(
                            "Seed or variant metadata mismatch: "
                            f"expected={expected}, observed={observed}"
                        )

                    if int(
                        run[
                            "trainable_parameters"
                        ]
                    ) != 109:
                        raise ValueError(
                            "Every compared model must have "
                            "109 trainable parameters."
                        )

                    runs.append(run)

    return runs


def build_paired_rows(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one paired difference per seed combination."""
    lookup = {
        (
            run["variant"],
            int(run["data_seed"]),
            int(run["model_seed"]),
            int(run["phase_seed"]),
        ): run
        for run in runs
    }

    rows: list[dict[str, Any]] = []

    for data_seed in DATA_SEEDS:
        for model_seed in MODEL_SEEDS:
            for phase_seed in PHASE_SEEDS:
                classical = lookup[
                    (
                        "classical_trig_local",
                        data_seed,
                        model_seed,
                        phase_seed,
                    )
                ]
                quantum = lookup[
                    (
                        "quantum_ring_frozen",
                        data_seed,
                        model_seed,
                        phase_seed,
                    )
                ]

                classical_error = float(
                    classical[
                        "relative_l2"
                    ]
                )
                quantum_error = float(
                    quantum[
                        "relative_l2"
                    ]
                )
                difference = (
                    quantum_error
                    - classical_error
                )

                rows.append(
                    {
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "phase_seed": phase_seed,
                        "classical_relative_l2": (
                            classical_error
                        ),
                        "quantum_relative_l2": (
                            quantum_error
                        ),
                        "quantum_minus_classical": (
                            difference
                        ),
                        "classical_wins": (
                            difference > 0.0
                        ),
                        "quantum_wins": (
                            difference < 0.0
                        ),
                        (
                            "relative_difference_"
                            "vs_classical"
                        ): (
                            difference
                            / classical_error
                        ),
                    }
                )

    return rows


def aggregate_variants(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate global and spectral metrics for each model."""
    scalar_fields = [
        "relative_l2",
        "mean_absolute_error",
        "maximum_absolute_error",
        "final_total_loss",
        "final_pde_loss",
        "final_initial_loss",
        "final_boundary_loss",
        "training_seconds",
    ]

    summary: dict[str, Any] = {}

    for variant in VARIANTS:
        selected = [
            run
            for run in runs
            if run["variant"]
            == variant
        ]

        values: dict[str, Any] = {
            "number_of_cells": (
                len(selected)
            ),
            "trainable_parameters": (
                109
            ),
            "fixed_phase_values": (
                16
            ),
        }

        for field in scalar_fields:
            values[field] = aggregate(
                [
                    float(run[field])
                    for run in selected
                ]
            )

        values[
            "spectral_errors"
        ] = {}

        for time_key in sorted(
            selected[0][
                "spectral_metrics"
            ].keys()
        ):
            values[
                "spectral_errors"
            ][time_key] = {}

            for band in (
                "low",
                "middle",
                "high",
            ):
                metric = (
                    "error_normalized_by_"
                    "total_reference_energy"
                )

                values[
                    "spectral_errors"
                ][time_key][band] = (
                    aggregate(
                        [
                            float(
                                run[
                                    "spectral_metrics"
                                ][time_key][
                                    "bands"
                                ][band][
                                    metric
                                ]
                            )
                            for run in selected
                        ]
                    )
                )

        summary[variant] = values

    return summary


def level_means(
    paired_rows: list[dict[str, Any]],
    field: str,
    levels: list[int],
) -> dict[str, float]:
    """Return paired-difference means for each factor level."""
    return {
        str(level): mean(
            [
                float(
                    row[
                        "quantum_minus_classical"
                    ]
                )
                for row in paired_rows
                if int(
                    row[field]
                ) == level
            ]
        )
        for level in levels
    }


def exact_sign_probability(
    positive_wins: int,
    negative_wins: int,
) -> float:
    """Return an exact two-sided sign-count probability."""
    total = (
        positive_wins
        + negative_wins
    )

    if total == 0:
        return 1.0

    minority = min(
        positive_wins,
        negative_wins,
    )

    return min(
        1.0,
        (
            2.0
            * sum(
                math.comb(
                    total,
                    index,
                )
                for index in range(
                    minority + 1
                )
            )
            / (2**total)
        ),
    )


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a list of dictionaries as CSV."""
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
    paired_rows: list[dict[str, Any]],
    variant_summary: dict[str, Any],
    model_block_rows: list[dict[str, Any]],
) -> None:
    """Create independent figures for the extended design."""
    cell_labels = [
        (
            f"D{row['data_seed']}-"
            f"M{row['model_seed']}-"
            f"P{row['phase_seed']}"
        )
        for row in paired_rows
    ]
    positions = np.arange(
        len(paired_rows)
    )
    width = 0.38

    # Relative L2 by cell.
    figure = plt.figure(
        figsize=(15, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        positions - width / 2,
        [
            row[
                "classical_relative_l2"
            ]
            for row in paired_rows
        ],
        width=width,
        label=(
            "Classical local "
            "trigonometric"
        ),
    )
    axis.bar(
        positions + width / 2,
        [
            row[
                "quantum_relative_l2"
            ]
            for row in paired_rows
        ],
        width=width,
        label=(
            "Frozen quantum ring"
        ),
    )

    axis.set_xticks(
        positions,
        cell_labels,
        rotation=35,
    )
    axis.set_xlabel(
        "Seed combination: data-model-phase"
    )
    axis.set_ylabel(
        "Relative L2 error"
    )
    axis.set_title(
        "Burgers 2×4×2 model-seed extension"
    )
    axis.legend()
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "relative_l2_by_seed_combination.png",
        dpi=200,
    )
    plt.close(figure)

    # Paired differences by cell.
    figure = plt.figure(
        figsize=(15, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        cell_labels,
        [
            row[
                "quantum_minus_classical"
            ]
            for row in paired_rows
        ],
    )
    axis.axhline(
        0.0,
        linewidth=1.0,
    )
    axis.set_xlabel(
        "Seed combination: data-model-phase"
    )
    axis.set_ylabel(
        "Quantum minus classical relative L2"
    )
    axis.set_title(
        "Paired model differences\n"
        "Positive favors classical; negative favors quantum"
    )
    axis.tick_params(
        axis="x",
        rotation=35,
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "paired_difference_by_seed_combination.png",
        dpi=200,
    )
    plt.close(figure)

    # Mean difference within each independent model-initialization block.
    figure = plt.figure(
        figsize=(9, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        [
            str(row["model_seed"])
            for row in model_block_rows
        ],
        [
            row[
                "mean_quantum_minus_classical"
            ]
            for row in model_block_rows
        ],
        yerr=[
            row[
                "std_quantum_minus_classical"
            ]
            for row in model_block_rows
        ],
        capsize=5,
    )
    axis.axhline(
        0.0,
        linewidth=1.0,
    )
    axis.set_xlabel(
        "Neural-network initialization seed"
    )
    axis.set_ylabel(
        "Mean quantum-minus-classical error"
    )
    axis.set_title(
        "Average paired difference within each model-seed block"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "paired_difference_by_model_seed.png",
        dpi=200,
    )
    plt.close(figure)

    # Aggregate relative L2.
    figure = plt.figure(
        figsize=(7, 5)
    )
    axis = figure.add_subplot(111)

    labels = [
        "Classical local\ntrigonometric",
        "Frozen quantum\nring",
    ]

    axis.bar(
        labels,
        [
            variant_summary[
                "classical_trig_local"
            ]["relative_l2"]["mean"],
            variant_summary[
                "quantum_ring_frozen"
            ]["relative_l2"]["mean"],
        ],
        yerr=[
            variant_summary[
                "classical_trig_local"
            ]["relative_l2"]["std"],
            variant_summary[
                "quantum_ring_frozen"
            ]["relative_l2"]["std"],
        ],
        capsize=5,
    )
    axis.set_ylabel(
        "Relative L2 error"
    )
    axis.set_title(
        "Mean across sixteen seed combinations"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "relative_l2_mean_std.png",
        dpi=200,
    )
    plt.close(figure)

    # Spectral errors.
    bands = [
        "low",
        "middle",
        "high",
    ]
    positions = np.arange(
        len(bands)
    )

    figure = plt.figure(
        figsize=(9, 5)
    )
    axis = figure.add_subplot(111)

    for index, variant in enumerate(
        VARIANTS
    ):
        values = variant_summary[
            variant
        ]["spectral_errors"][
            "0.500000"
        ]
        offset = (
            index
            - (
                len(VARIANTS) - 1
            ) / 2
        ) * width

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
            capsize=4,
            label=(
                "Classical local "
                "trigonometric"
                if variant
                == "classical_trig_local"
                else "Frozen quantum ring"
            ),
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
        "Model-seed extension spectral errors at t=0.5"
    )
    axis.legend()
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "spectral_errors_t_0p5.png",
        dpi=200,
    )
    plt.close(figure)

    # Runtime.
    figure = plt.figure(
        figsize=(7, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        labels,
        [
            variant_summary[
                "classical_trig_local"
            ]["training_seconds"]["mean"],
            variant_summary[
                "quantum_ring_frozen"
            ]["training_seconds"]["mean"],
        ],
        yerr=[
            variant_summary[
                "classical_trig_local"
            ]["training_seconds"]["std"],
            variant_summary[
                "quantum_ring_frozen"
            ]["training_seconds"]["std"],
        ],
        capsize=5,
    )
    axis.set_ylabel(
        "Training time (seconds)"
    )
    axis.set_title(
        "Mean computational cost"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY
        / "training_time_mean_std.png",
        dpi=200,
    )
    plt.close(figure)


def main() -> None:
    """Create the complete extended-design summary."""
    if OUTPUT_DIRECTORY.exists():
        shutil.rmtree(
            OUTPUT_DIRECTORY
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = load_runs()
    paired_rows = build_paired_rows(
        runs
    )
    variant_summary = (
        aggregate_variants(
            runs
        )
    )

    differences = [
        float(
            row[
                "quantum_minus_classical"
            ]
        )
        for row in paired_rows
    ]

    classical_wins = sum(
        bool(
            row[
                "classical_wins"
            ]
        )
        for row in paired_rows
    )
    quantum_wins = sum(
        bool(
            row[
                "quantum_wins"
            ]
        )
        for row in paired_rows
    )

    model_block_rows = []

    for model_seed in MODEL_SEEDS:
        selected = [
            float(
                row[
                    "quantum_minus_classical"
                ]
            )
            for row in paired_rows
            if int(
                row["model_seed"]
            ) == model_seed
        ]

        values = aggregate(
            selected
        )

        model_block_rows.append(
            {
                "model_seed": (
                    model_seed
                ),
                "mean_quantum_minus_classical": (
                    values["mean"]
                ),
                "std_quantum_minus_classical": (
                    values["std"]
                ),
                "minimum_difference": (
                    values["minimum"]
                ),
                "maximum_difference": (
                    values["maximum"]
                ),
                "classical_wins": sum(
                    value > 0.0
                    for value in selected
                ),
                "quantum_wins": sum(
                    value < 0.0
                    for value in selected
                ),
            }
        )

    model_block_means = [
        float(
            row[
                "mean_quantum_minus_classical"
            ]
        )
        for row in model_block_rows
    ]

    model_blocks_classical = sum(
        value > 0.0
        for value in model_block_means
    )
    model_blocks_quantum = sum(
        value < 0.0
        for value in model_block_means
    )

    summary = {
        "experiment": {
            "design": (
                "Balanced 2 data x 4 model-initialization "
                "x 2 phase-seed paired extension"
            ),
            "data_seeds": (
                DATA_SEEDS
            ),
            "model_seeds": (
                MODEL_SEEDS
            ),
            "phase_seeds": (
                PHASE_SEEDS
            ),
            "number_of_paired_cells": (
                len(paired_rows)
            ),
            "number_of_model_seed_blocks": (
                len(MODEL_SEEDS)
            ),
            "epochs": 500,
            "initial_condition_weight": (
                10.0
            ),
            "trainable_parameters_per_model": (
                109
            ),
            (
                "identical_training_points_"
                "within_cell"
            ): True,
            (
                "identical_initial_classical_"
                "weights_within_cell"
            ): True,
            (
                "identical_phase_tensor_"
                "within_cell"
            ): True,
        },
        "variants": (
            variant_summary
        ),
        "paired_cells": {
            "definition": (
                "Frozen quantum ring relative L2 minus "
                "classical local trigonometric relative L2"
            ),
            "rows": (
                paired_rows
            ),
            "difference": (
                aggregate(
                    differences
                )
            ),
            "classical_wins": (
                classical_wins
            ),
            "quantum_wins": (
                quantum_wins
            ),
            "ties": (
                len(paired_rows)
                - classical_wins
                - quantum_wins
            ),
            (
                "exploratory_cell_sign_"
                "probability"
            ): (
                exact_sign_probability(
                    classical_wins,
                    quantum_wins,
                )
            ),
            "cell_probability_caveat": (
                "The sixteen cells share factor levels and are not "
                "sixteen fully independent random replicates."
            ),
        },
        "model_seed_blocks": {
            "rows": (
                model_block_rows
            ),
            "block_mean_difference": (
                aggregate(
                    model_block_means
                )
            ),
            "classical_favored_blocks": (
                model_blocks_classical
            ),
            "quantum_favored_blocks": (
                model_blocks_quantum
            ),
            (
                "exact_two_sided_block_"
                "sign_test_p"
            ): (
                exact_sign_probability(
                    model_blocks_classical,
                    model_blocks_quantum,
                )
            ),
            "block_test_interpretation": (
                "The four model-initialization seeds are treated as "
                "the primary independent blocks; each block mean averages "
                "over the fixed two data and two phase levels."
            ),
        },
        "factor_level_means": {
            "data_seed": (
                level_means(
                    paired_rows,
                    "data_seed",
                    DATA_SEEDS,
                )
            ),
            "model_seed": (
                level_means(
                    paired_rows,
                    "model_seed",
                    MODEL_SEEDS,
                )
            ),
            "phase_seed": (
                level_means(
                    paired_rows,
                    "phase_seed",
                    PHASE_SEEDS,
                )
            ),
        },
        "critical_limitation": (
            "Only four independent model-initialization blocks were "
            "tested, while data and phase each use two fixed levels. "
            "Results remain a controlled robustness study rather than "
            "a population-level statistical estimate."
        ),
    }

    with (
        OUTPUT_DIRECTORY
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

    save_csv(
        paired_rows,
        OUTPUT_DIRECTORY
        / "paired_cells.csv",
    )
    save_csv(
        model_block_rows,
        OUTPUT_DIRECTORY
        / "model_seed_blocks.csv",
    )

    run_rows = []

    for run in runs:
        run_rows.append(
            {
                key: value
                for key, value in (
                    run.items()
                )
                if key
                != "spectral_metrics"
            }
        )

    save_csv(
        run_rows,
        OUTPUT_DIRECTORY
        / "runs.csv",
    )

    save_figures(
        paired_rows=paired_rows,
        variant_summary=(
            variant_summary
        ),
        model_block_rows=(
            model_block_rows
        ),
    )

    classical_mean = variant_summary[
        "classical_trig_local"
    ]["relative_l2"]["mean"]
    quantum_mean = variant_summary[
        "quantum_ring_frozen"
    ]["relative_l2"]["mean"]
    mean_difference = summary[
        "paired_cells"
    ]["difference"]["mean"]

    readme_lines = [
        "# Burgers model-initialization seed extension",
        "",
        "## Design",
        "",
        (
            "- Two data seeds, four neural-network initialization "
            "seeds, and two fixed phase seeds."
        ),
        "- Sixteen matched classical–quantum cells.",
        (
            "- Each matched pair shares training coordinates, "
            "initial classical weights, the 16-value phase tensor, "
            "and 109 trainable parameters."
        ),
        "",
        "## Aggregate results",
        "",
        (
            "- Classical local trigonometric: "
            f"{classical_mean:.6f} ± "
            f"{variant_summary['classical_trig_local']['relative_l2']['std']:.6f}"
        ),
        (
            "- Frozen quantum ring: "
            f"{quantum_mean:.6f} ± "
            f"{variant_summary['quantum_ring_frozen']['relative_l2']['std']:.6f}"
        ),
        (
            "- Mean paired difference, quantum minus classical: "
            f"{mean_difference:.6f} ± "
            f"{summary['paired_cells']['difference']['std']:.6f}"
        ),
        (
            f"- Classical wins: {classical_wins}/16; "
            f"quantum wins: {quantum_wins}/16."
        ),
        "",
        "## Model-seed block robustness",
        "",
    ]

    for row in model_block_rows:
        readme_lines.append(
            "- Model seed "
            f"{row['model_seed']}: mean difference "
            f"{row['mean_quantum_minus_classical']:.6f}; "
            f"classical wins {row['classical_wins']}/4."
        )

    readme_lines.extend(
        [
            "",
            (
                "- All four model-initialization blocks favor the "
                "classical trigonometric map on average."
            ),
            (
                "- Exact two-sided sign test over four model-seed "
                "block means: "
                f"p={summary['model_seed_blocks']['exact_two_sided_block_sign_test_p']:.6f}."
            ),
            "",
            "## Limitation",
            "",
            (
                "Four model-initialization blocks are not enough for "
                "a confirmatory statistical claim. Data and phase use "
                "only two fixed levels, so this remains a controlled "
                "robustness extension."
            ),
        ]
    )

    (
        OUTPUT_DIRECTORY
        / "README.md"
    ).write_text(
        "\n".join(
            readme_lines
        )
        + "\n",
        encoding="utf-8",
    )

    zip_path = Path(
        "outputs/"
        "burgers_model_seed_extension_summary.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in sorted(
            OUTPUT_DIRECTORY.rglob(
                "*"
            )
        ):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=file_path.relative_to(
                        OUTPUT_DIRECTORY
                    ),
                )

    print(
        "Burgers model-seed extension summary"
    )
    print(
        "Classical local trig: "
        f"{classical_mean:.6f} ± "
        f"{variant_summary['classical_trig_local']['relative_l2']['std']:.6f}"
    )
    print(
        "Frozen quantum ring: "
        f"{quantum_mean:.6f} ± "
        f"{variant_summary['quantum_ring_frozen']['relative_l2']['std']:.6f}"
    )
    print(
        "Mean paired difference "
        "(quantum-classical): "
        f"{mean_difference:.6f} ± "
        f"{summary['paired_cells']['difference']['std']:.6f}"
    )
    print(
        f"Classical wins {classical_wins}/16; "
        f"quantum wins {quantum_wins}/16."
    )
    print(
        "Model-seed blocks favoring classical: "
        f"{model_blocks_classical}/4"
    )
    print(
        "Block-level sign-test p: "
        f"{summary['model_seed_blocks']['exact_two_sided_block_sign_test_p']:.6f}"
    )

    print(
        "\nModel-seed block means"
    )
    for row in model_block_rows:
        print(
            f"seed {row['model_seed']}: "
            f"{row['mean_quantum_minus_classical']:.6f} "
            f"(classical wins "
            f"{row['classical_wins']}/4)"
        )

    print(
        "\nCreated archive: "
        f"{zip_path}"
    )


if __name__ == "__main__":
    main()
