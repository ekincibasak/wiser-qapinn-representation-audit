from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

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


@dataclass(frozen=True)
class QuantumAblationSpecification:
    """One improved-training quantum control."""

    name: str
    architecture: str
    constructor: Callable[[], HybridQuantumPINN]
    freeze_quantum: bool = False


def specifications() -> dict[str, QuantumAblationSpecification]:
    """Return the no-entanglement and frozen-ring controls."""
    return {
        "no_entanglement": QuantumAblationSpecification(
            name="qapinn_no_entanglement",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "4-qubit VQC(depth=2, no CNOT) -> "
                "4 <Z> -> 8 -> 1"
            ),
            constructor=lambda: HybridQuantumPINN(
                in_features=2,
                classical_width=8,
                n_qubits=4,
                quantum_depth=2,
                entanglement="none",
                out_features=1,
            ),
        ),
        "frozen_ring": QuantumAblationSpecification(
            name="qapinn_ring_frozen",
            architecture=(
                "2 -> 8 -> 4 angles -> "
                "frozen 4-qubit VQC(depth=2, ring CNOTs) -> "
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
            freeze_quantum=True,
        ),
    }


def freeze_quantum_layer(model: HybridQuantumPINN) -> None:
    """Freeze only the trainable variational-circuit angles."""
    model.quantum_layer.quantum_weights.requires_grad_(False)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def quantum_parameter_count(model: HybridQuantumPINN) -> int:
    """Count trainable quantum angles after any freezing."""
    parameter = model.quantum_layer.quantum_weights
    return parameter.numel() if parameter.requires_grad else 0


def run_variant(
    specification: QuantumAblationSpecification,
    seed: int,
    config: BurgersTrainingConfig,
    reference_path: Path,
    output_directory: Path,
) -> BurgersRunResult:
    """Train one matched quantum ablation for one seed."""
    reference = load_reference(reference_path)

    # Recreate exactly the same training coordinates used by the improved
    # classical and trainable-ring experiments.
    set_reproducible_seed(seed)
    shared_points = sample_burgers_training_points(
        config=config,
        dtype=torch.float32,
    )

    # Recreate deterministic model initialization for this seed.
    set_reproducible_seed(seed)
    model = specification.constructor()

    if specification.freeze_quantum:
        freeze_quantum_layer(model)

    # run_one_model expects a specification-like object. This small adapter
    # preserves its saving and evaluation logic.
    class Adapter:
        name = specification.name
        architecture = specification.architecture

        @staticmethod
        def constructor() -> nn.Module:
            return model

    bands = FrequencyBands(
        low_upper_fraction=0.20,
        middle_upper_fraction=0.50,
    )

    result = run_one_model(
        specification=Adapter,
        seed=seed,
        points=shared_points,
        config=config,
        reference=reference,
        bands=bands,
        output_directory=output_directory,
    )

    # Correct metadata after freezing because train_burgers.run_one_model
    # counts the circuit tensor size rather than requires_grad status.
    result.trainable_parameters = trainable_parameter_count(model)
    result.quantum_parameters = quantum_parameter_count(model)

    run_directory = (
        output_directory
        / f"seed_{seed}"
        / specification.name
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run improved Burgers quantum ablations with "
            "500 epochs and initial-condition weight 10."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "no_entanglement",
            "frozen_ring",
        ],
        default=[
            "no_entanglement",
            "frozen_ring",
        ],
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/burgers_reference.npz"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/improved_burgers_ablation_new"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    config = BurgersTrainingConfig(
        viscosity=0.01 / np.pi,
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

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    available = specifications()
    results = []

    for variant_name in arguments.variants:
        specification = available[variant_name]

        print(
            "\n=============================================="
        )
        print(
            f"Seed {arguments.seed} | {specification.name}"
        )
        print(
            "=============================================="
        )

        result = run_variant(
            specification=specification,
            seed=arguments.seed,
            config=config,
            reference_path=arguments.reference,
            output_directory=arguments.output_directory,
        )
        results.append(result)

        print(
            f"Completed {specification.name}: "
            f"relative L2={result.relative_l2:.6f}, "
            f"trainable parameters={result.trainable_parameters}, "
            f"time={result.training_seconds:.2f}s"
        )

    result_path = (
        arguments.output_directory
        / f"seed_{arguments.seed}"
        / "new_ablation_results.json"
    )
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
        )

    print(f"\nSaved seed results to: {result_path}")
