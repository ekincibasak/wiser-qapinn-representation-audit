# Figure Index and Provenance

The repository separates **quantitative result plots** from **conceptual explanatory graphics**.

## Quantitative result plots

These figures summarize actual experiment outputs stored in `results/` or the corresponding run summaries.

| File | Meaning |
|---|---|
| `figures/results/heat_relative_l2_ablation.png` | Heat four-model relative-L2 comparison |
| `figures/results/burgers_architecture_ablation.png` | Burgers architecture / classical-trigonometric comparison |
| `figures/results/burgers_robustness_16_cells.png` | Final 16 matched classical-vs-frozen-quantum cells |
| `figures/results/burgers_effective_rank.png` | Effective-rank comparison of primary representations |
| `figures/results/burgers_representation_cka.png` | Representation-similarity (linear CKA) matrix |
| `figures/results/burgers_spectral_errors_t_0p5.png` | Low/middle/high spectral errors at `t=0.5` |
| `figures/results/burgers_training_time.png` | Training-time comparison |

## Conceptual explanatory graphics

These figures explain the project architecture or research logic. They are **not experimental data**.

| File | Meaning |
|---|---|
| `figures/conceptual/graphical_summary.png` | End-to-end study logic |
| `figures/conceptual/central_research_question.png` | Attribution question and controls |
| `figures/conceptual/heat_vs_burgers.png` | Why the two PDE benchmarks are complementary |
| `figures/conceptual/classical_vs_qapinn.png` | High-level classical vs hybrid pathways |
| `figures/conceptual/actual_quantum_circuit.png` | Exact four-qubit circuit design used in the study |
| `figures/conceptual/future_roadmap.png` | Scientifically bounded future-work roadmap |

This separation is intentional: explanatory artwork should never be mistaken for numerical evidence.
