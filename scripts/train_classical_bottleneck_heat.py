from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from classical_bottleneck import (
    ClassicalBottleneckPINN,
    count_trainable_parameters,
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
    config = HeatTrainingConfig(
        epochs=500,
        learning_rate=1.0e-3,
        n_collocation=512,
        n_initial=128,
        n_boundary=128,
        seed=42,
        log_every=100,
    )

    set_reproducible_seed(config.seed)

    model = ClassicalBottleneckPINN(
        in_features=2,
        classical_width=8,
        bottleneck_dimension=4,
        out_features=1,
    )

    parameter_count = count_trainable_parameters(model)

    initial_parameters = [
        parameter.detach().cpu().clone()
        for parameter in model.parameters()
    ]

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

    assert history.total_loss[-1] < history.total_loss[0]
    assert change_norm > 0.0
    assert all(
        math.isfinite(value)
        for value in history.total_loss
    )

    output_directory = Path(
        "outputs/classical_bottleneck_heat_demo"
    )

    save_training_results(
        output_directory=output_directory,
        config=config,
        history=history,
        metrics=metrics,
        model=model,
    )

    summary = {
        "model": "ClassicalBottleneckPINN",
        "architecture": "2 -> 8 -> 4 -> 8 -> 1",
        "trainable_parameters": parameter_count,
        "initial_total_loss": history.total_loss[0],
        "final_total_loss": history.total_loss[-1],
        "final_pde_loss": history.pde_loss[-1],
        "final_initial_loss": history.initial_loss[-1],
        "final_boundary_loss": history.boundary_loss[-1],
        "parameter_change_norm": change_norm,
        **asdict(metrics),
    }

    with (output_directory / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nClassical bottleneck Heat training passed.")
    for key, value in summary.items():
        print(f"{key}: {value}")
