# Bundled Result Evidence

This directory contains compact CSV summaries used by the report and README.

- `final_benchmark_table.csv`: headline means, standard deviations, parameter counts, runtimes, and evidence levels.
- `heat_quantum_ablation_runs.csv`: per-seed Heat ablation runs.
- `burgers_architecture_ablation_runs.csv`: per-seed Burgers architecture and classical-trigonometric comparisons.
- `burgers_robustness_16_cells.csv`: the 16 matched classical-vs-frozen-quantum cells.
- `burgers_model_seed_blocks.csv`: four model-initialization block summaries.
- `burgers_robustness_runs.csv`: full run-level robustness output.
- `burgers_representation_summary.csv`: effective rank, redundancy, saturation, sensitivities, and linear-probe diagnostics.
- `burgers_mean_cka.csv`, `burgers_std_cka.csv`: representation similarity matrices.

Raw model checkpoints and every intermediate tensor are intentionally omitted. The scripts generate experiment outputs under `outputs/`.
