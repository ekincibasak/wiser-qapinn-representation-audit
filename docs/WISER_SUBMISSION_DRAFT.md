# WISER BQP Challenge Submission Draft

## Project title

**What Does the Quantum Layer Learn? An Explainable and Spectral Analysis of Quantum-Assisted Physics-Informed Neural Networks**

## 250-word summary

Physics-Informed Neural Networks (PINNs) solve differential equations by embedding physical residuals, initial conditions, and boundary conditions into the training loss. Quantum-assisted PINNs replace part of the classical network with a variational quantum circuit, but improved error alone does not show that the benefit is uniquely quantum. This project asks: after controlling trainable parameter count, training data, optimization budget, network initialization, and fixed phase values, what does the quantum layer actually learn, and can a comparable classical layer reproduce it?

I implemented classical and four-qubit quantum-assisted PINNs for the Heat and viscous Burgers equations using a differentiable PyTorch state-vector simulator. The study includes trainable and frozen quantum circuits, no-entanglement and ring-entanglement ablations, Fourier-band errors, gradient checks, runtime accounting, effective rank, feature correlation, input sensitivity, linear probes, and centered-kernel-alignment analysis.

For the Heat equation, the classical bottleneck was more accurate, while training the quantum angles produced almost no measurable benefit over a frozen circuit. For nonlinear Burgers, a frozen ring circuit initially improved over a basic classical bottleneck. However, a parameter-matched classical trigonometric feature map reproduced the quantum representation geometry and achieved better accuracy at much lower cost. In a controlled extension covering 16 matched combinations of data, network, and phase seeds, the classical map won 14 cases and all four model-initialization blocks on average.

The result is an explainable negative finding: periodic nonlinear inductive bias can help PDE learning, but the tested benefit is not uniquely quantum. The repository provides reproducible code, validated references, ablations, spectra, representation analyses, and honest limitations.

## Proposed solution

The project builds a controlled benchmark for Quantum-Assisted Physics-Informed Neural Networks rather than evaluating only one hybrid model. A classical PINN and a four-qubit QAPINN receive the same space-time coordinates and are trained through PDE, initial-condition, and boundary-condition losses. Heat provides an analytically solvable implementation control, while viscous Burgers provides nonlinear mode coupling and steep spatial structure.

The solution has four layers of evaluation. First, gradient and reference-solution tests verify that derivatives and physical residuals are correct. Second, parameter-matched model ablations compare independent qubits, ring-connected circuits, trainable circuits, and frozen quantum feature maps. Third, Fourier-band errors determine whether a model improves large-scale or fine-scale structure. Fourth, representation diagnostics—effective rank, feature correlation, CKA, sensitivity, and linear probes—explain how the middle layer transforms information.

The decisive classical control is a fixed trigonometric feature map with the same 109 trainable parameters and the same 16 phase values as the frozen quantum circuit. A separated-seed design independently controls training coordinates, classical initialization, and phase initialization. The final extension evaluates 16 matched seed combinations.

This approach answers not only “Which model is more accurate?” but also “What mechanism caused the difference?” It is designed to identify a genuine quantum-specific effect if one exists, while also producing a scientifically useful negative result when a simpler classical model reproduces the behavior.

## Why the project matters and where it can be applied

PINNs are used when physical equations are known but conventional simulations, inverse estimation, sparse observations, or repeated parameter studies are expensive. Potential application areas include fluid flow, thermal transport, diffusion, reaction systems, materials modeling, simplified magnetohydrodynamics, and inverse identification of physical coefficients.

The project’s immediate contribution is methodological: it provides a fair procedure for evaluating quantum layers inside physics-informed models. Without parameter matching, frozen-layer controls, spectral analysis, and classical nonlinear baselines, an apparent quantum improvement may actually come from compression, periodic activation functions, or initialization.

The current result shows that periodic nonlinear features can improve nonlinear PDE learning, but the tested advantage is not uniquely quantum. This is important because it redirects future research toward identifying the exact inductive biases that matter before moving to costly hardware experiments.

## Main results

- Heat: the 109-parameter classical bottleneck achieved \(0.6106\pm0.0579\) relative \(L_2\), compared with \(0.6501\pm0.0354\) for the trainable ring QAPINN.
- Burgers three-seed study: the classical local trigonometric map achieved \(0.4719\pm0.0348\), compared with \(0.5252\pm0.0655\) for the frozen ring circuit.
- Burgers 16-cell extension: the classical map achieved \(0.4475\pm0.0411\), compared with \(0.4887\pm0.0698\) for the frozen quantum ring.
- The classical control won 14/16 matched cells and all four model-initialization blocks on average.
- The classical and frozen quantum representations were highly similar, with CKA \(0.936\pm0.050\).
- The quantum simulation was approximately nine times slower in the final comparison.

## Technical and research limitations

The quantum circuit was evaluated with an exact ideal state-vector simulator. The experiments therefore do not include finite-shot uncertainty, gate noise, hardware topology, queue time, calibration drift, or measurement mitigation. A four-qubit circuit is also too small to support claims about scaling.

The Heat and early Burgers ablations use three seeds. The final robustness extension includes 16 matched cells, but only four neural-network initialization seeds can be treated as primary independent blocks. The block-level sign test is not significant at conventional thresholds. Data and phase factors use only two selected levels in the final extension.

The classical trigonometric layer is a strong functional control but not an exact symbolic classical simulation of the complete quantum circuit. High CKA indicates similar geometry, not identical causal mechanisms. The benchmark also uses fixed small networks and one Burgers viscosity; conclusions may change with larger models, other PDE regimes, or different circuit encodings.

Finally, the best errors remain imperfect. This project prioritizes controlled comparison and explanation rather than claiming state-of-the-art Burgers accuracy.

## Future testing, improvement, and expansion

The next step is to add more independent neural-network initialization blocks and several Burgers viscosities, especially lower-viscosity cases with sharper gradients. The study should also test trainable classical Fourier features, sinusoidal representation networks, and random Fourier features with matched computation and parameter counts.

For the quantum branch, future experiments should vary circuit depth, encoding repetition, measurement observables, placement of the quantum layer, and qubit count. Finite-shot simulation should be added before any hardware run. Small hardware experiments could then quantify whether noise changes representation rank, spectral error, or gradient stability.

The methodology can be expanded from Heat and Burgers to coupled advection–diffusion systems, reaction–diffusion equations, simplified MHD waves, thermal transport in materials, and inverse estimation of viscosity or diffusivity. A particularly useful extension would compare forward PDE solving with inverse problems, where sparse data and physically constrained latent features may change the relative value of the quantum map.

The repository will retain exact configurations, seed lists, reference data, compact result summaries, and scripts so that every conclusion can be reproduced or challenged.
