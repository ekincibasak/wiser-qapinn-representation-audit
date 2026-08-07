# Reproducibility Guide

## Environment

Recommended:

- Python 3.11 or newer;
- CPU execution is sufficient for all four-qubit state-vector experiments;
- a clean virtual environment;
- PyTorch, NumPy, Matplotlib, and pytest.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Deterministic controls

The final Burgers matched comparisons separate:

- `data_seed`: samples PDE, initial, and boundary points;
- `model_seed`: initializes the surrounding classical network;
- `phase_seed`: generates the shared 16-value phase tensor.

Within each paired cell:

1. training coordinates are identical;
2. classical initial weights are identical;
3. phase tensors are identical;
4. both models have 109 trainable parameters;
5. the same optimizer, epoch budget, loss weights, reference grid, and spectral bands are used.

## Reference solutions

### Heat

The exact solution is evaluated analytically.

### Burgers

`burgers_reference.npz` contains a \(401\times101\) Cole–Hopf reference grid. The integral is evaluated with stabilized Gauss–Hermite quadrature. The source implementation validates:

- the manufactured autograd residual;
- initial conditions;
- zero boundary conditions;
- quadrature convergence.

## Suggested execution order

```bash
pytest

python -m scripts.compare_quantum_ablations_heat

python -m scripts.train_burgers_improved

python -m scripts.compare_burgers_quantum_ablations_improved

python -m scripts.compare_classical_trigonometric_burgers_v2
```

For one matched separated-seed pair:

```bash
python -m scripts.run_burgers_factorial_pair \
  --data-seed 11 \
  --model-seed 777 \
  --phase-seed 101 \
  --output-root outputs/factorial
```

## Result files

Compact result summaries are under `results/`. Raw model states and all historical prediction arrays are intentionally omitted from this final GitHub package to keep it small. The experiment scripts regenerate them.

## Interpretation rules

- Compare dense-grid error, not only sampled training loss.
- Report mean and standard deviation.
- Keep paired seeds aligned.
- Do not call a simulator result hardware performance.
- Do not call a lower pilot error quantum advantage.
- Distinguish trainable parameters from fixed feature-map constants.
- Treat sign tests and factorial effects according to their actual independent replication level.
