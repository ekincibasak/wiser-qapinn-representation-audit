# How to Read the Main Result

## What the models actually did

The PINN/QAPINN approximates the PDE solution `u_theta(x,t)` by minimizing the PDE residual together with initial- and boundary-condition losses.

For viscous Burgers,

```text
u_t + u u_x = nu u_xx
```

and the learned residual is

```text
r_theta = u_t + u_theta u_x - nu u_xx.
```

The **Cole-Hopf transformation was not used by the neural network to learn the solution**. It was used to generate an independent reference solution, evaluated numerically with stabilized Gauss-Hermite quadrature, against which dense-grid prediction error was measured.

The **Fourier analysis was also not the PDE solver**. It was a diagnostic: prediction errors were decomposed into low-, middle-, and high-spatial-frequency bands to determine which spatial scales each architecture represented well.

## Why a better QAPINN than the plain bottleneck is not enough for quantum advantage

On the three-seed Burgers architecture study:

- classical bottleneck: relative L2 `0.6500 +/- 0.0831`;
- frozen quantum ring: `0.5252 +/- 0.0655`.

Thus the ring-based QAPINN was better than this **particular plain classical baseline**.

However, changing the middle layer changes more than whether the computation is quantum: it also changes periodicity, nonlinearity, feature geometry, and inductive bias. To test whether the gain was uniquely quantum, a parameter-matched classical trigonometric middle layer was added.

That stronger control achieved:

- classical local trigonometric: `0.4719 +/- 0.0348`.

Its output geometry was also highly similar to the frozen quantum output (`CKA = 0.936 +/- 0.050`). In the final 16 matched robustness cells, the classical control won 14/16 cells and had lower average error.

The evidence therefore supports the statement:

> The tested quantum layer exposes a useful periodic nonlinear inductive bias, but the observed gain over a plain bottleneck cannot be attributed to a robust quantum-specific mechanism.

This is a **model-attribution result**, not a hardware or computational quantum-advantage claim.
