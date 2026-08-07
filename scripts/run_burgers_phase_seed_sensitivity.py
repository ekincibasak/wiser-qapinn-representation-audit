from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from classical_trigonometric import ClassicalTrigonometricPINN
from quantum_layer import HybridQuantumPINN
from spectra import FrequencyBands
from train_burgers import (
    BurgersTrainingConfig,
    evaluate_model,
    load_reference,
    sample_burgers_training_points,
    train_model,
)
from train_heat import set_reproducible_seed


Variant = Literal[
    "classical_trig_local",
    "quantum_ring_frozen",
]


@dataclass
class SeparatedSeedRun:
    """One run with independently controlled data, model, and phase seeds."""

    variant: str
    data_seed: int
    model_seed: int
    phase_seed: int

    trainable_parameters: int
    fixed_phase_values: int

    initial_total_loss: float
    final_total_loss: float
    final_pde_loss: float
    final_initial_loss: float
    final_boundary_loss: float

    relative_l2: float
    mean_absolute_error: float
    maximum_absolute_error: float

    training_seconds: float
    spectral_metrics: dict[str, Any]


def make_phase_tensor(
    phase_seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the shared 16-value phase tensor independently."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(phase_seed)

    phases = torch.empty((2, 4, 2), dtype=dtype)
    phases.uniform_(-0.10, 0.10, generator=generator)
    return phases


def build_models(
    model_seed: int,
    phase_seed: int,
) -> tuple[ClassicalTrigonometricPINN, HybridQuantumPINN, float]:
    """Build matched classical and frozen-quantum models.

    The surrounding classical weights are initialized with model_seed.
    The middle phase tensor is then overwritten using phase_seed.
    """
    shared_phases = make_phase_tensor(
        phase_seed=phase_seed,
        dtype=torch.float32,
    )

    set_reproducible_seed(model_seed)
    classical_model = ClassicalTrigonometricPINN(
        in_features=2,
        classical_width=8,
        feature_dimension=4,
        feature_depth=2,
        connectivity="local",
        out_features=1,
    )

    set_reproducible_seed(model_seed)
    quantum_model = HybridQuantumPINN(
        in_features=2,
        classical_width=8,
        n_qubits=4,
        quantum_depth=2,
        entanglement="ring",
        out_features=1,
    )

    with torch.no_grad():
        classical_model.feature_map.fixed_phases.copy_(shared_phases)
        quantum_model.quantum_layer.quantum_weights.copy_(shared_phases)

    quantum_model.quantum_layer.quantum_weights.requires_grad_(False)

    classical_layers = [
        "input_layer",
        "angle_projection",
        "expansion_layer",
        "output_layer",
    ]

    maximum_classical_weight_difference = 0.0

    for layer_name in classical_layers:
        classical_layer = getattr(classical_model, layer_name)
        quantum_layer = getattr(quantum_model, layer_name)

        weight_difference = torch.max(
            torch.abs(
                classical_layer.weight.detach()
                - quantum_layer.weight.detach()
            )
        ).item()
        bias_difference = torch.max(
            torch.abs(
                classical_layer.bias.detach()
                - quantum_layer.bias.detach()
            )
        ).item()

        maximum_classical_weight_difference = max(
            maximum_classical_weight_difference,
            weight_difference,
            bias_difference,
        )

    phase_difference = torch.max(
        torch.abs(
            classical_model.feature_map.fixed_phases
            - quantum_model.quantum_layer.quantum_weights
        )
    ).item()

    if maximum_classical_weight_difference > 0.0:
        raise RuntimeError(
            "Classical network initializations do not match. "
            f"Maximum difference={maximum_classical_weight_difference}."
        )

    if phase_difference > 0.0:
        raise RuntimeError(
            "Fixed phase tensors do not match. "
            f"Maximum difference={phase_difference}."
        )

    return (
        classical_model,
        quantum_model,
        maximum_classical_weight_difference,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters updated by Adam."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_variant(
    variant: Variant,
    data_seed: int,
    model_seed: int,
    phase_seed: int,
    output_root: Path,
) -> SeparatedSeedRun:
    """Train and evaluate one independently seeded variant."""
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
        log_every=100,
    )

    bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    set_reproducible_seed(data_seed)
    shared_points = sample_burgers_training_points(
        config=config,
        dtype=torch.float32,
    )

    (
        classical_model,
        quantum_model,
        maximum_initial_weight_difference,
    ) = build_models(
        model_seed=model_seed,
        phase_seed=phase_seed,
    )

    if variant == "classical_trig_local":
        model: nn.Module = classical_model
    elif variant == "quantum_ring_frozen":
        model = quantum_model
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    training = train_model(
        model=model,
        points=shared_points,
        config=config,
    )

    evaluation = evaluate_model(
        model=model,
        reference=reference,
        bands=bands,
    )

    run = SeparatedSeedRun(
        variant=variant,
        data_seed=data_seed,
        model_seed=model_seed,
        phase_seed=phase_seed,
        trainable_parameters=count_trainable_parameters(model),
        fixed_phase_values=16,
        initial_total_loss=training["history"]["total"][0],
        final_total_loss=training["history"]["total"][-1],
        final_pde_loss=training["history"]["pde"][-1],
        final_initial_loss=training["history"]["initial"][-1],
        final_boundary_loss=training["history"]["boundary"][-1],
        relative_l2=evaluation["relative_l2"],
        mean_absolute_error=evaluation["mean_absolute_error"],
        maximum_absolute_error=evaluation["maximum_absolute_error"],
        training_seconds=training["training_seconds"],
        spectral_metrics=evaluation["spectral_metrics"],
    )

    run_directory = (
        output_root
        / (
            f"data_{data_seed}"
            f"_model_{model_seed}"
            f"_phase_{phase_seed}"
        )
        / variant
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        run_directory / "model_state.pt",
    )
    np.save(
        run_directory / "prediction.npy",
        evaluation["prediction"],
    )

    with (run_directory / "history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training["history"],
            file,
            indent=2,
        )

    with (run_directory / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                **asdict(run),
                "maximum_initial_classical_weight_difference": (
                    maximum_initial_weight_difference
                ),
                "shared_training_points": True,
                "phase_tensor_shared_between_variants": True,
            },
            file,
            indent=2,
        )

    return run


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a Burgers phase-seed sensitivity experiment "
            "with independently controlled data, model, and phase seeds."
        )
    )
    parser.add_argument(
        "--variant",
        choices=[
            "classical_trig_local",
            "quantum_ring_frozen",
        ],
        required=True,
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=777,
    )
    parser.add_argument(
        "--phase-seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/burgers_phase_seed_sensitivity"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    start = time.perf_counter()

    result = run_variant(
        variant=arguments.variant,
        data_seed=arguments.data_seed,
        model_seed=arguments.model_seed,
        phase_seed=arguments.phase_seed,
        output_root=arguments.output_root,
    )

    print("\nSeparated-seed Burgers run completed")
    print(f"variant={result.variant}")
    print(
        f"data_seed={result.data_seed}, "
        f"model_seed={result.model_seed}, "
        f"phase_seed={result.phase_seed}"
    )
    print(
        f"trainable_parameters="
        f"{result.trainable_parameters}"
    )
    print(f"relative_L2={result.relative_l2:.6f}")
    print(f"final_PDE={result.final_pde_loss:.6f}")
    print(f"final_IC={result.final_initial_loss:.6f}")
    print(f"final_BC={result.final_boundary_loss:.6f}")
    print(
        f"training_seconds="
        f"{result.training_seconds:.2f}"
    )
    print(
        "wall_clock_seconds="
        f"{time.perf_counter() - start:.2f}"
    )
