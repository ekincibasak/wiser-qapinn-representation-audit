import math

import numpy as np
import torch

from burgers import burgers_initial_condition, cole_hopf_burgers_solution
from classical_bottleneck import ClassicalBottleneckPINN
from classical_trigonometric import ClassicalTrigonometricPINN
from heat import ExactHeatSolutionModel, heat_equation_residual
from quantum_layer import HybridQuantumPINN, count_trainable_parameters


def test_parameter_counts_and_shapes():
    classical = ClassicalBottleneckPINN()
    trig = ClassicalTrigonometricPINN(connectivity="local")
    quantum = HybridQuantumPINN(entanglement="ring")

    assert sum(p.numel() for p in classical.parameters() if p.requires_grad) == 109
    assert sum(p.numel() for p in trig.parameters() if p.requires_grad) == 109
    assert count_trainable_parameters(quantum) == 125
    assert quantum.quantum_layer.quantum_weights.numel() == 16

    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    assert classical(x).shape == (2, 1)
    assert trig(x).shape == (2, 1)
    assert quantum(x).shape == (2, 1)


def test_quantum_measurements_are_bounded_and_differentiable():
    torch.manual_seed(7)
    model = HybridQuantumPINN(entanglement="ring")
    xt = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float32, requires_grad=True)

    encoded = model.encode(xt)
    z = encoded.quantum_measurements
    assert torch.all(z <= 1.0 + 1e-10)
    assert torch.all(z >= -1.0 - 1e-10)

    loss = encoded.prediction.square().mean()
    grad = torch.autograd.grad(loss, model.quantum_layer.quantum_weights)[0]
    assert torch.all(torch.isfinite(grad))
    assert torch.linalg.vector_norm(grad).item() > 0.0


def test_exact_heat_solution_has_negligible_residual():
    model = ExactHeatSolutionModel(alpha=0.1).to(dtype=torch.float64)
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).reshape(-1, 1)
    result = heat_equation_residual(model, x, t, alpha=0.1)
    assert torch.max(torch.abs(result.residual)).item() < 1e-10


def test_cole_hopf_reference_matches_initial_condition():
    x = np.linspace(-1.0, 1.0, 41)
    u0 = cole_hopf_burgers_solution(x=x, t=0.0, viscosity=0.01 / math.pi)
    expected = -np.sin(math.pi * x)
    assert np.max(np.abs(u0 - expected)) < 1e-12

    x_torch = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
    assert torch.max(torch.abs(burgers_initial_condition(x_torch).flatten() - torch.tensor(expected))).item() < 1e-12
