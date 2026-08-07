"""Regenerate compact final-result plots from committed CSV summaries.

This script does not rerun neural-network training; it visualizes values already
stored under results/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "results" / "regenerated"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_bar(labels, means, stds, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4)
    ax.set_xticks(positions, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def main() -> None:
    bench = read_csv(RESULTS / "final_benchmark_table.csv")

    heat = [row for row in bench if row["benchmark"].startswith("Heat")]
    save_bar(
        [row["model"] for row in heat],
        [float(row["relative_l2_mean"]) for row in heat],
        [float(row["relative_l2_std"]) for row in heat],
        "Heat: controlled four-model comparison",
        "Relative L2 error",
        "heat_relative_l2.png",
    )

    burgers3 = [
        row for row in bench
        if row["benchmark"] == "Burgers, 500 epochs, 3 seeds"
    ]
    save_bar(
        [row["model"] for row in burgers3],
        [float(row["relative_l2_mean"]) for row in burgers3],
        [float(row["relative_l2_std"]) for row in burgers3],
        "Burgers: architecture comparison",
        "Relative L2 error",
        "burgers_architecture_relative_l2.png",
    )

    paired = read_csv(RESULTS / "burgers_robustness_16_cells.csv")
    labels = [
        f"d{row['data_seed']}-m{row['model_seed']}-p{row['phase_seed']}"
        for row in paired
    ]
    classical = np.array([float(row["classical_relative_l2"]) for row in paired])
    quantum = np.array([float(row["quantum_relative_l2"]) for row in paired])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(paired))
    width = 0.38
    ax.bar(x - width / 2, classical, width, label="Classical local trig")
    ax.bar(x + width / 2, quantum, width, label="Frozen quantum ring")
    ax.set_xticks(x, labels, rotation=70, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Burgers: 16 matched robustness cells")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "burgers_16_matched_cells.png", dpi=220)
    plt.close(fig)

    rep_rows = read_csv(RESULTS / "burgers_representation_summary.csv")
    chosen_names = {
        "classical_bottleneck",
        "trig_local_features",
        "quantum_ring_frozen_measurements",
    }
    selected = [row for row in rep_rows if row["representation"] in chosen_names]
    save_bar(
        [row["display_name"] for row in selected],
        [float(row["effective_rank_entropy_mean"]) for row in selected],
        [float(row["effective_rank_entropy_std"]) for row in selected],
        "Burgers: effective rank of primary representations",
        "Effective rank",
        "burgers_effective_rank.png",
    )

    print(f"Regenerated summary plots in {OUT}")


if __name__ == "__main__":
    main()
