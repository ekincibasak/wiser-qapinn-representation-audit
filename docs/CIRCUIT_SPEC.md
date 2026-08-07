# Actual Quantum Circuit Used

The reported QAPINN uses a **four-qubit, depth-2, exact state-vector circuit**.

## Classical preprocessing

The coordinates are mapped to four angles:

```text
(x,t) -> Linear(2,8) -> tanh -> Linear(8,4) -> tanh -> (phi_0,...,phi_3)
```

## Data encoding

Each qubit starts in `|0>` and receives one data-encoding rotation:

```text
q0: RY(phi_0)
q1: RY(phi_1)
q2: RY(phi_2)
q3: RY(phi_3)
```

## Variational block

For each of two depths, every qubit receives

```text
RY(theta[l,i]) -> RZ(omega[l,i])
```

followed by the selected entanglement pattern.

The trainable circuit therefore contains

```text
2 depths x 4 qubits x 2 rotation parameters = 16 quantum parameters.
```

## Main variants

- **No-entanglement QAPINN:** no CNOTs.
- **Trainable ring QAPINN:** CNOTs `(0->1), (1->2), (2->3), (3->0)`; the 16 rotation angles are trainable.
- **Frozen ring QAPINN:** the same ring circuit, but its 16 quantum angles are fixed during PINN training.

## Measurement and return to the classical network

The circuit returns four expectation values

```text
<Z0>, <Z1>, <Z2>, <Z3>
```

which are processed as

```text
4 -> Linear(4,8) -> tanh -> Linear(8,1) -> u_theta(x,t).
```

## Backend

The implementation is a custom differentiable PyTorch state-vector simulator. The reported experiments do **not** use quantum hardware, finite-shot sampling, a device-noise model, or an external quantum SDK/framework.

See `figures/conceptual/actual_quantum_circuit.png`.
