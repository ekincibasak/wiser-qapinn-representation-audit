# Repository Validation

The rebuilt repository was checked before packaging.

## Automated tests

```text
4 passed
```

The tests cover:

- controlled model parameter counts (109 classical, 125 trainable QAPINN, 16 quantum angles);
- quantum measurement bounds and gradient connectivity;
- exact Heat-equation residual consistency;
- Cole-Hopf Burgers reference consistency at the initial condition.

## CLI/import checks

The main experiment entry points were imported and their command-line interfaces were checked with `--help`.

## Training smoke test

A one-seed, five-epoch Heat ablation was run successfully across:

- classical bottleneck;
- QAPINN without entanglement;
- trainable ring QAPINN;
- frozen ring QAPINN.

This smoke test is an implementation check only; it is not part of the reported benchmark evidence.
