# Final Benchmark Summary

## Evidence levels

The tables deliberately separate:

1. **implementation validation** — derivative checks, exact/reference solutions, and gradient connectivity;
2. **controlled pilots** — small seed sets used to identify architectural behavior;
3. **robustness extensions** — matched variation of data, network, and phase seeds.

A lower error in a pilot is not treated as a quantum advantage claim.

## Heat equation

Training protocol: 100 epochs, 128 collocation points, 32 initial points, 32 boundary points, seeds 11/22/33.

| Model | Trainable | Relative \(L_2\) | Mean time |
|---|---:|---:|---:|
| **Classical bottleneck** | **109** | **0.6106 ± 0.0579** | **0.27 s** |
| Quantum, no entanglement | 125 | 0.6605 ± 0.0284 | 5.09 s |
| Quantum, trainable ring | 125 | 0.6501 ± 0.0354 | 4.44 s |
| Quantum, frozen ring | 109 | 0.6504 ± 0.0310 | 5.36 s |

Interpretation:

- the classical bottleneck was best;
- ring entanglement was only slightly better than independent qubits;
- trainable versus frozen ring error differed by about \(2.5\times10^{-4}\);
- the trained and frozen output geometries had CKA approximately 0.999;
- the Heat ring output had effective rank near 1.10, indicating strong representation compression.

## Burgers equation: three-seed architecture study

Training protocol: 500 epochs, \(w_{IC}=10\), seeds 11/22/33.

| Model | Trainable | Relative \(L_2\) | Mean time |
|---|---:|---:|---:|
| Classical bottleneck | 109 | 0.6500 ± 0.0831 | 2.04 s |
| **Classical local trigonometric** | **109** | **0.4719 ± 0.0348** | **2.08 s** |
| Classical ring trigonometric | 109 | 0.5857 ± 0.0597 | 2.28 s |
| Quantum, no entanglement | 125 | 0.5911 ± 0.0141 | 25.42 s |
| Quantum, trainable ring | 125 | 0.5339 ± 0.0485 | 25.06 s |
| Quantum, frozen ring | 109 | 0.5252 ± 0.0655 | 18.26 s |

The frozen ring circuit improved over an ordinary bottleneck, but the local classical trigonometric control improved further.

## Burgers equation: 16-cell matched robustness extension

Design:

\[
2\,\text{ data seeds}\times4\,\text{ model seeds}\times2\,\text{ phase seeds}=16\,\text{ matched cells}.
\]

Each classical–quantum pair shared training points, surrounding classical initialization, phase values, optimizer, budget, and 109 trainable parameters.

| Metric | Classical local trig | Frozen quantum ring |
|---|---:|---:|
| Relative \(L_2\) | **0.4475 ± 0.0411** | 0.4887 ± 0.0698 |
| MAE | **0.1839 ± 0.0251** | 0.2112 ± 0.0413 |
| PDE loss | 0.2355 ± 0.0390 | **0.2139 ± 0.0639** |
| IC loss | **0.0235 ± 0.0107** | 0.0376 ± 0.0269 |
| BC loss | **0.0897 ± 0.0450** | 0.1183 ± 0.0563 |
| Time | **2.99 ± 1.43 s** | 26.58 ± 5.14 s |

At \(t=0.5\):

| Spectral band | Classical local trig | Frozen quantum ring |
|---|---:|---:|
| Low | **0.51985 ± 0.03844** | 0.55401 ± 0.06474 |
| Middle | **0.05485 ± 0.00198** | 0.05593 ± 0.00223 |
| High | **0.01431 ± 0.00348** | 0.01612 ± 0.00355 |

The classical map won 14/16 cells. Every one of the four network-initialization blocks favored it on average. The exact two-sided block sign test was \(p=0.125\), so this remains robustness evidence rather than formal statistical confirmation.

## Representation analysis

| Representation | Effective rank | Mean absolute correlation |
|---|---:|---:|
| Classical bottleneck | 1.390 ± 0.308 | 0.731 |
| **Classical local trig** | **2.079 ± 0.229** | **0.532** |
| Frozen quantum ring | 1.602 ± 0.473 | 0.685 |

The local classical and frozen quantum outputs had CKA:

\[
0.936\pm0.050.
\]

The classical control therefore recreated much of the quantum geometry while retaining more independent directions.

## Final conclusion

The experiments support a **periodic nonlinear inductive-bias effect**, not a demonstrated quantum-specific advantage.
