from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from quantum_layer import (
    HybridQuantumPINN,
    count_trainable_parameters,
    quantum_parameter_count,
)
from train_heat import (
    HeatTrainingConfig,
    evaluate_heat_pinn,
    parameter_change_norm,
    save_training_results,
    set_reproducible_seed,
    train_heat_pinn,
)


if __name__ == "__main__":
    # This is deliberately a small integration test.
    # The final classical-vs-quantum experiment must use matched
    # training points, seeds, epochs, and compute budgets.
    config = HeatTrainingConfig(
        epochs=50,
        learning_rate=1.0e-3,
        n_collocation=128,
        n_initial=32,
        n_boundary=32,
        seed=42,
        log_every=10,
    )

    set_reproducible_seed(config.seed)

    model = HybridQuantumPINN(
        in_features=2,
        classical_width=8,
        n_qubits=4,
        quantum_depth=2,
        entanglement="ring",
        out_features=1,
    )

    initial_parameters = [
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    ]

    initial_quantum_weights = (
        model.quantum_layer.quantum_weights
        .detach()
        .cpu()
        .clone()
    )

    history = train_heat_pinn(
        model=model,
        config=config,
        device="cpu",
    )

    metrics = evaluate_heat_pinn(
        model=model,
        alpha=config.alpha,
        device="cpu",
        n_x=101,
        n_t=51,
    )

    total_change = parameter_change_norm(
        initial_parameters=initial_parameters,
        model=model,
    )

    quantum_change = torch.linalg.vector_norm(
        model.quantum_layer.quantum_weights
        .detach()
        .cpu()
        - initial_quantum_weights
    ).item()

    assert history.total_loss[-1] < history.total_loss[0]
    assert total_change > 0.0
    assert quantum_change > 0.0

    output_directory = Path(
        "outputs/quantum_heat_smoke_test"
    )

    save_training_results(
        output_directory=output_directory,
        config=config,
        history=history,
        metrics=metrics,
        model=model,
    )

    summary = {
        "purpose": (
            "Integration smoke test only; not a fair final "
            "classical-versus-quantum comparison."
        ),
        "backend": "ideal PyTorch state-vector simulator",
        "n_qubits": model.n_qubits,
        "quantum_depth": model.quantum_depth,
        "entanglement": model.entanglement,
        "total_trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "quantum_trainable_parameters": (
            quantum_parameter_count(model)
        ),
        "initial_total_loss": history.total_loss[0],
        "final_total_loss": history.total_loss[-1],
        "final_pde_loss": history.pde_loss[-1],
        "final_initial_loss": history.initial_loss[-1],
        "final_boundary_loss": history.boundary_loss[-1],
        "total_parameter_change_norm": total_change,
        "quantum_parameter_change_norm": quantum_change,
        **asdict(metrics),
    }

    with (output_directory / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nQuantum Heat integration smoke test passed.")
    for key, value in summary.items():
        print(f"{key}: {value}")
