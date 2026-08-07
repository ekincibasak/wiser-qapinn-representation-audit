from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from classical_bottleneck import ClassicalBottleneckPINN
from classical_trigonometric import ClassicalTrigonometricPINN
from quantum_layer import HybridQuantumPINN


@dataclass(frozen=True)
class RepresentationSpecification:
    """One trained model and one internal four-dimensional representation."""

    name: str
    display_name: str
    model_name: str
    representation_name: str
    model_constructor: Callable[[], nn.Module]


@dataclass
class RepresentationMetrics:
    """Diagnostics for one representation and one seed."""

    representation: str
    display_name: str
    model: str
    seed: int
    dimension: int
    number_of_points: int

    mean: list[float]
    standard_deviation: list[float]
    minimum: list[float]
    maximum: list[float]

    saturation_fraction_abs_ge_0p95: float
    mean_absolute_off_diagonal_correlation: float

    singular_values: list[float]
    singular_value_energy_fractions: list[float]
    effective_rank_entropy: float
    participation_ratio: float

    rms_sensitivity_to_x: float
    rms_sensitivity_to_t: float
    sensitivity_ratio_x_over_t: float | None

    linear_probe_r2_x: float
    linear_probe_r2_t: float


MODEL_ROOT = Path(
    "outputs/burgers_classical_trig_control_v2"
)


def representation_specifications() -> list[RepresentationSpecification]:
    """Return the angle inputs and four-value outputs of all middle layers."""
    return [
        RepresentationSpecification(
            name="classical_bottleneck",
            display_name="Classical bottleneck",
            model_name="classical_bottleneck",
            representation_name="bottleneck",
            model_constructor=lambda: ClassicalBottleneckPINN(
                in_features=2,
                classical_width=8,
                bottleneck_dimension=4,
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="trig_local_angles",
            display_name="Local trig: input angles",
            model_name="classical_trig_local",
            representation_name="encoded_angles",
            model_constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="local",
            ),
        ),
        RepresentationSpecification(
            name="trig_local_features",
            display_name="Local trig: output features",
            model_name="classical_trig_local",
            representation_name="trigonometric_features",
            model_constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="local",
            ),
        ),
        RepresentationSpecification(
            name="trig_ring_angles",
            display_name="Ring trig: input angles",
            model_name="classical_trig_ring",
            representation_name="encoded_angles",
            model_constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="ring",
            ),
        ),
        RepresentationSpecification(
            name="trig_ring_features",
            display_name="Ring trig: output features",
            model_name="classical_trig_ring",
            representation_name="trigonometric_features",
            model_constructor=lambda: ClassicalTrigonometricPINN(
                connectivity="ring",
            ),
        ),
        RepresentationSpecification(
            name="quantum_no_ent_angles",
            display_name="Quantum local: input angles",
            model_name="qapinn_no_entanglement",
            representation_name="encoded_angles",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="none",
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="quantum_no_ent_measurements",
            display_name="Quantum local: measurements",
            model_name="qapinn_no_entanglement",
            representation_name="quantum_measurements",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="none",
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="quantum_ring_trainable_angles",
            display_name="Trainable ring: input angles",
            model_name="qapinn_ring_trainable",
            representation_name="encoded_angles",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="quantum_ring_trainable_measurements",
            display_name="Trainable ring: measurements",
            model_name="qapinn_ring_trainable",
            representation_name="quantum_measurements",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="quantum_ring_frozen_angles",
            display_name="Frozen ring: input angles",
            model_name="qapinn_ring_frozen",
            representation_name="encoded_angles",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
        RepresentationSpecification(
            name="quantum_ring_frozen_measurements",
            display_name="Frozen ring: measurements",
            model_name="qapinn_ring_frozen",
            representation_name="quantum_measurements",
            model_constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="ring",
                out_features=1,
            ),
        ),
    ]


def primary_output_names() -> list[str]:
    """Return the output representations used for the main comparison."""
    return [
        "classical_bottleneck",
        "trig_local_features",
        "trig_ring_features",
        "quantum_no_ent_measurements",
        "quantum_ring_trainable_measurements",
        "quantum_ring_frozen_measurements",
    ]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch state dictionary using the safer API when available."""
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


def build_grid(
    n_x: int,
    n_t: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Create the Burgers grid x in [-1,1], t in [0,1]."""
    if n_x < 2 or n_t < 2:
        raise ValueError("n_x and n_t must be at least 2.")

    x = torch.linspace(
        -1.0,
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

    inputs = torch.stack(
        [
            x_grid.reshape(-1),
            t_grid.reshape(-1),
        ],
        dim=1,
    )

    return (
        inputs,
        x_grid.reshape(-1).cpu().numpy(),
        t_grid.reshape(-1).cpu().numpy(),
    )


def extract_representation(
    model: nn.Module,
    inputs: torch.Tensor,
    representation_name: str,
) -> torch.Tensor:
    """Extract one named four-dimensional representation."""
    if isinstance(model, ClassicalBottleneckPINN):
        outputs = model.encode(inputs)

        if representation_name != "bottleneck":
            raise ValueError(
                "ClassicalBottleneckPINN only exposes 'bottleneck'."
            )

        representation = outputs.bottleneck

    elif isinstance(model, ClassicalTrigonometricPINN):
        outputs = model.encode(inputs)

        if representation_name == "encoded_angles":
            representation = outputs.encoded_angles
        elif representation_name == "trigonometric_features":
            representation = outputs.trigonometric_features
        else:
            raise ValueError(
                "ClassicalTrigonometricPINN supports 'encoded_angles' "
                "and 'trigonometric_features'."
            )

    elif isinstance(model, HybridQuantumPINN):
        outputs = model.encode(inputs)

        if representation_name == "encoded_angles":
            representation = outputs.encoded_angles
        elif representation_name == "quantum_measurements":
            representation = outputs.quantum_measurements
        else:
            raise ValueError(
                "HybridQuantumPINN supports 'encoded_angles' and "
                "'quantum_measurements'."
            )

    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}."
        )

    if representation.ndim != 2 or representation.shape[1] != 4:
        raise ValueError(
            "Expected representation shape (number_of_points, 4), "
            f"received {tuple(representation.shape)}."
        )

    return representation


def stable_correlation_matrix(
    values: np.ndarray,
) -> np.ndarray:
    """Return a finite feature-correlation matrix."""
    centered = values - np.mean(
        values,
        axis=0,
        keepdims=True,
    )
    standard_deviation = np.std(
        centered,
        axis=0,
        ddof=0,
        keepdims=True,
    )

    standardized = np.divide(
        centered,
        standard_deviation,
        out=np.zeros_like(centered),
        where=standard_deviation > 1.0e-12,
    )

    correlation = (
        standardized.T @ standardized
    ) / values.shape[0]

    diagonal = np.where(
        standard_deviation.reshape(-1) > 1.0e-12,
        1.0,
        0.0,
    )
    np.fill_diagonal(
        correlation,
        diagonal,
    )

    return correlation


def singular_value_metrics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return singular values, energy fractions, and two rank measures."""
    centered = values - np.mean(
        values,
        axis=0,
        keepdims=True,
    )

    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
        full_matrices=False,
    )

    energy = singular_values**2
    total_energy = float(
        np.sum(energy)
    )

    if total_energy <= 1.0e-20:
        return (
            singular_values,
            np.zeros_like(energy),
            0.0,
            0.0,
        )

    fractions = energy / total_energy

    positive = fractions[
        fractions > 1.0e-15
    ]
    entropy = -float(
        np.sum(
            positive * np.log(positive)
        )
    )

    effective_rank = float(
        np.exp(entropy)
    )
    participation_ratio = float(
        1.0 / np.sum(fractions**2)
    )

    return (
        singular_values,
        fractions,
        effective_rank,
        participation_ratio,
    )


def calculate_sensitivities(
    model: nn.Module,
    inputs: torch.Tensor,
    representation_name: str,
) -> tuple[float, float]:
    """Calculate RMS latent derivatives with respect to x and t."""
    differentiable_inputs = (
        inputs.clone()
        .detach()
        .requires_grad_(True)
    )

    representation = extract_representation(
        model=model,
        inputs=differentiable_inputs,
        representation_name=representation_name,
    )

    gradients = []

    for feature_index in range(4):
        gradient = torch.autograd.grad(
            outputs=representation[
                :,
                feature_index,
            ].sum(),
            inputs=differentiable_inputs,
            create_graph=False,
            retain_graph=(
                feature_index < 3
            ),
        )[0]
        gradients.append(
            gradient.detach()
        )

    jacobian = torch.stack(
        gradients,
        dim=1,
    )

    rms_x = float(
        torch.sqrt(
            torch.mean(
                jacobian[:, :, 0] ** 2
            )
        ).cpu()
    )
    rms_t = float(
        torch.sqrt(
            torch.mean(
                jacobian[:, :, 1] ** 2
            )
        ).cpu()
    )

    return rms_x, rms_t


def r2_score(
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return the coefficient of determination."""
    residual = float(
        np.sum(
            (target - prediction) ** 2
        )
    )
    total = float(
        np.sum(
            (
                target
                - np.mean(target)
            ) ** 2
        )
    )

    if total <= 1.0e-20:
        return 0.0

    return 1.0 - residual / total


def linear_probe_scores(
    representation: np.ndarray,
    x_values: np.ndarray,
    t_values: np.ndarray,
    seed: int,
    ridge: float = 1.0e-6,
) -> tuple[float, float]:
    """Test held-out linear recovery of x and t."""
    generator = np.random.default_rng(
        seed + 2026
    )
    indices = generator.permutation(
        representation.shape[0]
    )
    split = int(
        0.8 * representation.shape[0]
    )

    train_indices = indices[:split]
    test_indices = indices[split:]

    train_values = representation[
        train_indices
    ]
    test_values = representation[
        test_indices
    ]

    train_mean = np.mean(
        train_values,
        axis=0,
        keepdims=True,
    )
    train_std = np.std(
        train_values,
        axis=0,
        keepdims=True,
    )

    standardized_train = np.divide(
        train_values - train_mean,
        train_std,
        out=np.zeros_like(train_values),
        where=train_std > 1.0e-12,
    )
    standardized_test = np.divide(
        test_values - train_mean,
        train_std,
        out=np.zeros_like(test_values),
        where=train_std > 1.0e-12,
    )

    design_train = np.concatenate(
        [
            np.ones(
                (
                    standardized_train.shape[0],
                    1,
                )
            ),
            standardized_train,
        ],
        axis=1,
    )
    design_test = np.concatenate(
        [
            np.ones(
                (
                    standardized_test.shape[0],
                    1,
                )
            ),
            standardized_test,
        ],
        axis=1,
    )

    penalty = ridge * np.eye(
        design_train.shape[1]
    )
    penalty[0, 0] = 0.0

    inverse = np.linalg.pinv(
        design_train.T @ design_train
        + penalty
    )

    def fit_and_score(
        target: np.ndarray,
    ) -> float:
        coefficients = (
            inverse
            @ design_train.T
            @ target[train_indices]
        )
        prediction = (
            design_test @ coefficients
        )

        return r2_score(
            target[test_indices],
            prediction,
        )

    return (
        fit_and_score(x_values),
        fit_and_score(t_values),
    )


def linear_cka(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return linear centered kernel alignment."""
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            "CKA inputs must contain the same number of points."
        )

    first_centered = first - np.mean(
        first,
        axis=0,
        keepdims=True,
    )
    second_centered = second - np.mean(
        second,
        axis=0,
        keepdims=True,
    )

    cross = (
        first_centered.T
        @ second_centered
    )
    first_gram = (
        first_centered.T
        @ first_centered
    )
    second_gram = (
        second_centered.T
        @ second_centered
    )

    numerator = float(
        np.sum(cross**2)
    )
    denominator = math.sqrt(
        float(np.sum(first_gram**2))
        * float(np.sum(second_gram**2))
    )

    if denominator <= 1.0e-20:
        return 0.0

    return numerator / denominator


def analyze_one(
    specification: RepresentationSpecification,
    seed: int,
    inputs: torch.Tensor,
    x_values: np.ndarray,
    t_values: np.ndarray,
) -> tuple[
    RepresentationMetrics,
    np.ndarray,
    np.ndarray,
]:
    """Load one trained model and analyze one internal representation."""
    model = specification.model_constructor().to(
        dtype=inputs.dtype
    )

    state_path = (
        MODEL_ROOT
        / f"seed_{seed}"
        / specification.model_name
        / "model_state.pt"
    )

    if not state_path.exists():
        raise FileNotFoundError(
            f"Missing trained state: {state_path}"
        )

    model.load_state_dict(
        load_state_dict(state_path)
    )
    model.eval()

    with torch.no_grad():
        representation_tensor = extract_representation(
            model=model,
            inputs=inputs,
            representation_name=(
                specification.representation_name
            ),
        )

    representation = (
        representation_tensor
        .cpu()
        .numpy()
    )

    correlation = stable_correlation_matrix(
        representation
    )
    off_diagonal_mask = ~np.eye(
        4,
        dtype=bool,
    )
    mean_absolute_correlation = float(
        np.mean(
            np.abs(
                correlation[
                    off_diagonal_mask
                ]
            )
        )
    )

    (
        singular_values,
        energy_fractions,
        effective_rank,
        participation_ratio,
    ) = singular_value_metrics(
        representation
    )

    rms_x, rms_t = calculate_sensitivities(
        model=model,
        inputs=inputs,
        representation_name=(
            specification.representation_name
        ),
    )

    r2_x, r2_t = linear_probe_scores(
        representation=representation,
        x_values=x_values,
        t_values=t_values,
        seed=seed,
    )

    metrics = RepresentationMetrics(
        representation=specification.name,
        display_name=specification.display_name,
        model=specification.model_name,
        seed=seed,
        dimension=4,
        number_of_points=representation.shape[0],
        mean=np.mean(
            representation,
            axis=0,
        ).tolist(),
        standard_deviation=np.std(
            representation,
            axis=0,
            ddof=0,
        ).tolist(),
        minimum=np.min(
            representation,
            axis=0,
        ).tolist(),
        maximum=np.max(
            representation,
            axis=0,
        ).tolist(),
        saturation_fraction_abs_ge_0p95=float(
            np.mean(
                np.abs(representation)
                >= 0.95
            )
        ),
        mean_absolute_off_diagonal_correlation=(
            mean_absolute_correlation
        ),
        singular_values=singular_values.tolist(),
        singular_value_energy_fractions=(
            energy_fractions.tolist()
        ),
        effective_rank_entropy=effective_rank,
        participation_ratio=participation_ratio,
        rms_sensitivity_to_x=rms_x,
        rms_sensitivity_to_t=rms_t,
        sensitivity_ratio_x_over_t=(
            rms_x / rms_t
            if rms_t > 1.0e-15
            else None
        ),
        linear_probe_r2_x=r2_x,
        linear_probe_r2_t=r2_t,
    )

    return (
        metrics,
        representation,
        correlation,
    )


def aggregate_scalar(
    selected: list[RepresentationMetrics],
    field: str,
) -> dict[str, float]:
    """Aggregate one scalar metric across seeds."""
    values = [
        float(
            getattr(
                item,
                field,
            )
        )
        for item in selected
    ]

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
    }


def summarize(
    all_metrics: list[RepresentationMetrics],
) -> dict[str, Any]:
    """Build representation-wise summaries."""
    summary: dict[str, Any] = {}

    for specification in representation_specifications():
        selected = [
            item
            for item in all_metrics
            if item.representation
            == specification.name
        ]

        summary[
            specification.name
        ] = {
            "display_name": (
                specification.display_name
            ),
            "model": (
                specification.model_name
            ),
            "representation_name": (
                specification.representation_name
            ),
            "effective_rank_entropy": (
                aggregate_scalar(
                    selected,
                    "effective_rank_entropy",
                )
            ),
            "participation_ratio": (
                aggregate_scalar(
                    selected,
                    "participation_ratio",
                )
            ),
            (
                "mean_absolute_off_diagonal_correlation"
            ): (
                aggregate_scalar(
                    selected,
                    "mean_absolute_off_diagonal_correlation",
                )
            ),
            (
                "saturation_fraction_abs_ge_0p95"
            ): (
                aggregate_scalar(
                    selected,
                    "saturation_fraction_abs_ge_0p95",
                )
            ),
            "rms_sensitivity_to_x": (
                aggregate_scalar(
                    selected,
                    "rms_sensitivity_to_x",
                )
            ),
            "rms_sensitivity_to_t": (
                aggregate_scalar(
                    selected,
                    "rms_sensitivity_to_t",
                )
            ),
            "linear_probe_r2_x": (
                aggregate_scalar(
                    selected,
                    "linear_probe_r2_x",
                )
            ),
            "linear_probe_r2_t": (
                aggregate_scalar(
                    selected,
                    "linear_probe_r2_t",
                )
            ),
        }

    return summary


def save_summary_csv(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save aggregate metrics as CSV."""
    metric_names = [
        "effective_rank_entropy",
        "participation_ratio",
        "mean_absolute_off_diagonal_correlation",
        "saturation_fraction_abs_ge_0p95",
        "rms_sensitivity_to_x",
        "rms_sensitivity_to_t",
        "linear_probe_r2_x",
        "linear_probe_r2_t",
    ]

    rows = []

    for name, values in summary.items():
        row: dict[str, Any] = {
            "representation": name,
            "display_name": values[
                "display_name"
            ],
            "model": values["model"],
            "representation_name": values[
                "representation_name"
            ],
        }

        for metric_name in metric_names:
            row[
                f"{metric_name}_mean"
            ] = values[
                metric_name
            ]["mean"]
            row[
                f"{metric_name}_std"
            ] = values[
                metric_name
            ]["std"]

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


def save_matrix_csv(
    labels: list[str],
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Save a labeled square matrix."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "representation",
                *labels,
            ]
        )

        for label, row in zip(
            labels,
            matrix,
            strict=True,
        ):
            writer.writerow(
                [
                    label,
                    *row.tolist(),
                ]
            )


def save_bar_plot(
    summary: dict[str, Any],
    names: list[str],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save one aggregate bar chart."""
    labels = [
        summary[name][
            "display_name"
        ]
        for name in names
    ]
    means = [
        summary[name][
            metric_name
        ]["mean"]
        for name in names
    ]
    deviations = [
        summary[name][
            metric_name
        ]["std"]
        for name in names
    ]

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        labels,
        means,
        yerr=deviations,
        capsize=4,
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(
        axis="x",
        rotation=22,
    )
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


def save_sensitivity_plot(
    summary: dict[str, Any],
    names: list[str],
    output_path: Path,
) -> None:
    """Save x- and t-sensitivity for the primary outputs."""
    labels = [
        summary[name][
            "display_name"
        ]
        for name in names
    ]
    positions = np.arange(
        len(names)
    )
    width = 0.36

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        positions - width / 2,
        [
            summary[name][
                "rms_sensitivity_to_x"
            ]["mean"]
            for name in names
        ],
        width=width,
        yerr=[
            summary[name][
                "rms_sensitivity_to_x"
            ]["std"]
            for name in names
        ],
        capsize=4,
        label="Sensitivity to x",
    )
    axis.bar(
        positions + width / 2,
        [
            summary[name][
                "rms_sensitivity_to_t"
            ]["mean"]
            for name in names
        ],
        width=width,
        yerr=[
            summary[name][
                "rms_sensitivity_to_t"
            ]["std"]
            for name in names
        ],
        capsize=4,
        label="Sensitivity to t",
    )

    axis.set_xticks(
        positions,
        labels,
        rotation=22,
    )
    axis.set_ylabel(
        "RMS representation derivative"
    )
    axis.set_title(
        "Input sensitivity of improved Burgers representations"
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


def save_linear_probe_plot(
    summary: dict[str, Any],
    names: list[str],
    output_path: Path,
) -> None:
    """Save held-out linear decoding scores."""
    labels = [
        summary[name][
            "display_name"
        ]
        for name in names
    ]
    positions = np.arange(
        len(names)
    )
    width = 0.36

    figure = plt.figure(
        figsize=(12, 5)
    )
    axis = figure.add_subplot(111)

    axis.bar(
        positions - width / 2,
        [
            summary[name][
                "linear_probe_r2_x"
            ]["mean"]
            for name in names
        ],
        width=width,
        label="Decode x",
    )
    axis.bar(
        positions + width / 2,
        [
            summary[name][
                "linear_probe_r2_t"
            ]["mean"]
            for name in names
        ],
        width=width,
        label="Decode t",
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )
    axis.set_xticks(
        positions,
        labels,
        rotation=22,
    )
    axis.set_ylabel(
        "Held-out linear-probe R²"
    )
    axis.set_title(
        "Input information retained by improved Burgers representations"
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


def save_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    value_min: float | None = None,
    value_max: float | None = None,
) -> None:
    """Save one heatmap with numeric annotations."""
    figure = plt.figure(
        figsize=(10, 9)
    )
    axis = figure.add_subplot(111)

    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=value_min,
        vmax=value_max,
    )

    axis.set_xticks(
        np.arange(
            len(labels)
        ),
        labels,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        np.arange(
            len(labels)
        ),
        labels,
    )
    axis.set_title(title)

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(
        colorbar_label
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)


def run_self_check() -> None:
    """Validate CKA and effective-rank calculations."""
    generator = np.random.default_rng(
        123
    )
    values = generator.normal(
        size=(200, 4)
    )

    if not np.isclose(
        linear_cka(
            values,
            values,
        ),
        1.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "CKA self-similarity check failed."
        )

    rank_one = np.column_stack(
        [
            values[:, 0],
            2.0 * values[:, 0],
            -values[:, 0],
            0.5 * values[:, 0],
        ]
    )
    (
        _,
        _,
        effective_rank,
        participation_ratio,
    ) = singular_value_metrics(
        rank_one
    )

    if not np.isclose(
        effective_rank,
        1.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Effective-rank self-check failed."
        )

    if not np.isclose(
        participation_ratio,
        1.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Participation-ratio self-check failed."
        )

    print(
        "Improved Burgers representation self-check passed."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze improved Burgers classical and quantum "
            "four-dimensional internal representations."
        )
    )
    parser.add_argument(
        "--n-x",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--n-t",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/improved_burgers_representation_analysis"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_self_check()

    arguments = parse_arguments()

    (
        inputs,
        x_values,
        t_values,
    ) = build_grid(
        n_x=arguments.n_x,
        n_t=arguments.n_t,
        dtype=torch.float64,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    specifications = (
        representation_specifications()
    )
    labels = [
        specification.name
        for specification in specifications
    ]
    display_labels = [
        specification.display_name
        for specification in specifications
    ]

    all_metrics: list[
        RepresentationMetrics
    ] = []
    seed_cka_matrices = []
    correlation_accumulator: dict[
        str,
        list[np.ndarray],
    ] = {
        label: []
        for label in labels
    }

    for seed in (
        11,
        22,
        33,
    ):
        print(
            "\n========================================"
        )
        print(
            f"Analyzing improved Burgers seed {seed}"
        )
        print(
            "========================================"
        )

        seed_directory = (
            arguments.output_directory
            / f"seed_{seed}"
        )
        seed_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        seed_representations: dict[
            str,
            np.ndarray,
        ] = {}

        for specification in specifications:
            (
                metrics,
                representation,
                correlation,
            ) = analyze_one(
                specification=specification,
                seed=seed,
                inputs=inputs,
                x_values=x_values,
                t_values=t_values,
            )

            all_metrics.append(
                metrics
            )
            seed_representations[
                specification.name
            ] = representation
            correlation_accumulator[
                specification.name
            ].append(correlation)

            np.save(
                seed_directory
                / (
                    f"{specification.name}"
                    "_values.npy"
                ),
                representation,
            )
            np.save(
                seed_directory
                / (
                    f"{specification.name}"
                    "_correlation.npy"
                ),
                correlation,
            )

            print(
                f"{specification.name:40s} | "
                f"rank={metrics.effective_rank_entropy:.4f} | "
                f"corr={metrics.mean_absolute_off_diagonal_correlation:.4f} | "
                f"R2(x)={metrics.linear_probe_r2_x:.4f} | "
                f"R2(t)={metrics.linear_probe_r2_t:.4f}"
            )

        cka_matrix = np.empty(
            (
                len(labels),
                len(labels),
            ),
            dtype=float,
        )

        for first_index, first_label in enumerate(
            labels
        ):
            for second_index, second_label in enumerate(
                labels
            ):
                cka_matrix[
                    first_index,
                    second_index,
                ] = linear_cka(
                    seed_representations[
                        first_label
                    ],
                    seed_representations[
                        second_label
                    ],
                )

        seed_cka_matrices.append(
            cka_matrix
        )
        np.save(
            seed_directory
            / "cka_matrix.npy",
            cka_matrix,
        )
        save_matrix_csv(
            labels=labels,
            matrix=cka_matrix,
            output_path=(
                seed_directory
                / "cka_matrix.csv"
            ),
        )

    summary = summarize(
        all_metrics
    )

    stacked_cka = np.stack(
        seed_cka_matrices,
        axis=0,
    )
    mean_cka = np.mean(
        stacked_cka,
        axis=0,
    )
    std_cka = np.std(
        stacked_cka,
        axis=0,
        ddof=1,
    )

    with (
        arguments.output_directory
        / "metrics_by_seed.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                asdict(item)
                for item in all_metrics
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
            {
                "analysis_grid": {
                    "x_min": -1.0,
                    "x_max": 1.0,
                    "t_min": 0.0,
                    "t_max": 1.0,
                    "n_x": arguments.n_x,
                    "n_t": arguments.n_t,
                    "number_of_points": (
                        arguments.n_x
                        * arguments.n_t
                    ),
                },
                "seeds": [
                    11,
                    22,
                    33,
                ],
                "primary_outputs": (
                    primary_output_names()
                ),
                "representations": (
                    summary
                ),
                "cka_labels": labels,
                "cka_display_labels": (
                    display_labels
                ),
                "mean_cka": (
                    mean_cka.tolist()
                ),
                "std_cka": (
                    std_cka.tolist()
                ),
            },
            file,
            indent=2,
        )

    save_summary_csv(
        summary=summary,
        output_path=(
            arguments.output_directory
            / "summary.csv"
        ),
    )
    save_matrix_csv(
        labels=labels,
        matrix=mean_cka,
        output_path=(
            arguments.output_directory
            / "mean_cka.csv"
        ),
    )
    save_matrix_csv(
        labels=labels,
        matrix=std_cka,
        output_path=(
            arguments.output_directory
            / "std_cka.csv"
        ),
    )

    primary_names = (
        primary_output_names()
    )

    save_bar_plot(
        summary=summary,
        names=primary_names,
        metric_name=(
            "effective_rank_entropy"
        ),
        ylabel=(
            "Effective rank (maximum 4)"
        ),
        title=(
            "Effective dimensionality of improved Burgers outputs"
        ),
        output_path=(
            arguments.output_directory
            / "effective_rank_primary_outputs.png"
        ),
    )
    save_bar_plot(
        summary=summary,
        names=primary_names,
        metric_name=(
            "mean_absolute_off_diagonal_correlation"
        ),
        ylabel=(
            "Mean absolute off-diagonal correlation"
        ),
        title=(
            "Feature redundancy of improved Burgers outputs"
        ),
        output_path=(
            arguments.output_directory
            / "feature_redundancy_primary_outputs.png"
        ),
    )
    save_bar_plot(
        summary=summary,
        names=primary_names,
        metric_name=(
            "saturation_fraction_abs_ge_0p95"
        ),
        ylabel=(
            "Fraction with |value| ≥ 0.95"
        ),
        title=(
            "Saturation of improved Burgers outputs"
        ),
        output_path=(
            arguments.output_directory
            / "saturation_primary_outputs.png"
        ),
    )
    save_sensitivity_plot(
        summary=summary,
        names=primary_names,
        output_path=(
            arguments.output_directory
            / "input_sensitivity_primary_outputs.png"
        ),
    )
    save_linear_probe_plot(
        summary=summary,
        names=primary_names,
        output_path=(
            arguments.output_directory
            / "linear_probe_primary_outputs.png"
        ),
    )
    save_heatmap(
        matrix=mean_cka,
        labels=display_labels,
        title=(
            "Mean linear CKA across improved Burgers representations"
        ),
        colorbar_label="Linear CKA",
        output_path=(
            arguments.output_directory
            / "mean_cka_all_representations.png"
        ),
        value_min=0.0,
        value_max=1.0,
    )

    primary_indices = [
        labels.index(name)
        for name in primary_names
    ]
    primary_cka = mean_cka[
        np.ix_(
            primary_indices,
            primary_indices,
        )
    ]
    primary_display = [
        summary[name][
            "display_name"
        ]
        for name in primary_names
    ]

    save_heatmap(
        matrix=primary_cka,
        labels=primary_display,
        title=(
            "Mean CKA among improved Burgers middle-layer outputs"
        ),
        colorbar_label="Linear CKA",
        output_path=(
            arguments.output_directory
            / "mean_cka_primary_outputs.png"
        ),
        value_min=0.0,
        value_max=1.0,
    )

    transformation_pairs = {
        "trig_local_input_to_output": (
            "trig_local_angles",
            "trig_local_features",
        ),
        "trig_ring_input_to_output": (
            "trig_ring_angles",
            "trig_ring_features",
        ),
        "quantum_local_input_to_output": (
            "quantum_no_ent_angles",
            "quantum_no_ent_measurements",
        ),
        "quantum_trainable_ring_input_to_output": (
            "quantum_ring_trainable_angles",
            "quantum_ring_trainable_measurements",
        ),
        "quantum_frozen_ring_input_to_output": (
            "quantum_ring_frozen_angles",
            "quantum_ring_frozen_measurements",
        ),
        "local_trig_output_vs_frozen_quantum_output": (
            "trig_local_features",
            "quantum_ring_frozen_measurements",
        ),
        "local_trig_output_vs_trainable_quantum_output": (
            "trig_local_features",
            "quantum_ring_trainable_measurements",
        ),
        "trig_ring_output_vs_frozen_quantum_output": (
            "trig_ring_features",
            "quantum_ring_frozen_measurements",
        ),
    }

    selected_cka: dict[str, Any] = {}

    for comparison_name, (
        first_name,
        second_name,
    ) in transformation_pairs.items():
        first_index = labels.index(
            first_name
        )
        second_index = labels.index(
            second_name
        )

        values = [
            float(
                matrix[
                    first_index,
                    second_index,
                ]
            )
            for matrix in seed_cka_matrices
        ]

        selected_cka[
            comparison_name
        ] = {
            "first": first_name,
            "second": second_name,
            "mean": mean(values),
            "std": (
                stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "by_seed": {
                str(seed): value
                for seed, value in zip(
                    (
                        11,
                        22,
                        33,
                    ),
                    values,
                    strict=True,
                )
            },
        }

    with (
        arguments.output_directory
        / "selected_cka_comparisons.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected_cka,
            file,
            indent=2,
        )

    mean_correlations = {
        name: np.mean(
            np.stack(
                matrices,
                axis=0,
            ),
            axis=0,
        )
        for name, matrices in (
            correlation_accumulator.items()
        )
    }

    for name in primary_names:
        save_heatmap(
            matrix=mean_correlations[
                name
            ],
            labels=[
                "feature 1",
                "feature 2",
                "feature 3",
                "feature 4",
            ],
            title=(
                "Mean feature correlation: "
                f"{summary[name]['display_name']}"
            ),
            colorbar_label=(
                "Correlation"
            ),
            output_path=(
                arguments.output_directory
                / (
                    "correlation_"
                    f"{name}.png"
                )
            ),
            value_min=-1.0,
            value_max=1.0,
        )

    lines = [
        "# Improved Burgers internal-representation analysis",
        "",
        "This analysis compares the trained four-dimensional middle-layer "
        "representations from the 500-epoch, initial-condition-weight-10 "
        "Burgers experiments.",
        "",
        "## Main outputs",
        "",
    ]

    for name in primary_names:
        values = summary[name]

        lines.extend(
            [
                f"### {values['display_name']}",
                "",
                (
                    "- Effective rank: "
                    f"{values['effective_rank_entropy']['mean']:.6f} ± "
                    f"{values['effective_rank_entropy']['std']:.6f}"
                ),
                (
                    "- Mean absolute feature correlation: "
                    f"{values['mean_absolute_off_diagonal_correlation']['mean']:.6f}"
                ),
                (
                    "- Linear probe R2(x): "
                    f"{values['linear_probe_r2_x']['mean']:.6f}"
                ),
                (
                    "- Linear probe R2(t): "
                    f"{values['linear_probe_r2_t']['mean']:.6f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Selected CKA comparisons",
            "",
        ]
    )

    for comparison_name, values in (
        selected_cka.items()
    ):
        lines.append(
            f"- {comparison_name}: "
            f"{values['mean']:.6f} ± "
            f"{values['std']:.6f}"
        )

    (
        arguments.output_directory
        / "README.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\nAggregate primary-output representation results"
    )

    for name in primary_names:
        values = summary[name]

        print(
            f"{name:42s} | "
            f"rank="
            f"{values['effective_rank_entropy']['mean']:.4f} ± "
            f"{values['effective_rank_entropy']['std']:.4f} | "
            f"corr="
            f"{values['mean_absolute_off_diagonal_correlation']['mean']:.4f} | "
            f"R2(x)="
            f"{values['linear_probe_r2_x']['mean']:.4f} | "
            f"R2(t)="
            f"{values['linear_probe_r2_t']['mean']:.4f}"
        )

    print(
        "\nSelected mean CKA comparisons"
    )

    for comparison_name, values in (
        selected_cka.items()
    ):
        print(
            f"{comparison_name:52s} | "
            f"{values['mean']:.6f} ± "
            f"{values['std']:.6f}"
        )

    print(
        "\nSaved outputs to: "
        f"{arguments.output_directory}"
    )
