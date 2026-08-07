# Repository Map

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── quantum_layer.py                 # exact 4-qubit PyTorch state-vector layer
├── classical_bottleneck.py          # 109-parameter plain bottleneck control
├── classical_trigonometric.py       # 109-parameter periodic classical control
├── heat.py                          # Heat PDE and exact solution
├── burgers.py                       # Burgers PDE + Cole-Hopf reference
├── derivatives.py                   # automatic derivatives
├── losses.py                        # PDE/IC/BC losses
├── spectra.py                       # Fourier/spectral diagnostics
├── train_heat.py
├── train_burgers.py
├── data/
│   └── burgers_reference.npz
├── scripts/                         # experiment and analysis entry points
├── tests/                           # implementation validation tests
├── results/                         # compact numerical evidence
├── figures/
│   ├── results/                     # quantitative plots
│   └── conceptual/                  # explanatory figures
├── docs/
│   ├── WISER_BQP_Technical_Report.pdf
│   ├── CIRCUIT_SPEC.md
│   ├── SCIENTIFIC_INTERPRETATION.md
│   ├── FINAL_BENCHMARKS.md
│   ├── EXPERIMENT_MATRIX.md
│   └── REPRODUCIBILITY.md
└── outputs/                         # generated files; ignored by git
```
