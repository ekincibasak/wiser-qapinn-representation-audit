from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn


EntanglementPattern = Literal["none", "linear", "ring"]


@dataclass
class QuantumPINNOutputs:
    """Intermediate values produced by the hybrid quantum PINN."""

    classical_features: torch.Tensor
    encoded_angles: torch.Tensor
    quantum_measurements: torch.Tensor
    post_quantum_features: torch.Tensor
    prediction: torch.Tensor


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype matching a real floating-point dtype."""
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128

    raise TypeError(
        "The quantum simulator supports torch.float32 and torch.float64."
    )


def _ry_gate(
    angles: torch.Tensor,
) -> torch.Tensor:
    """Return batched RY rotation matrices.

    For each angle phi,

        RY(phi) =
        [[cos(phi/2), -sin(phi/2)],
         [sin(phi/2),  cos(phi/2)]].
    """
    complex_dtype = _complex_dtype(angles.dtype)

    cosine = torch.cos(angles / 2.0)
    sine = torch.sin(angles / 2.0)

    row_0 = torch.stack([cosine, -sine], dim=-1)
    row_1 = torch.stack([sine, cosine], dim=-1)

    return torch.stack(
        [row_0, row_1],
        dim=-2,
    ).to(dtype=complex_dtype)


def _rz_gate(
    angles: torch.Tensor,
) -> torch.Tensor:
    """Return batched RZ rotation matrices.

    For each angle phi,

        RZ(phi) =
        diag(exp(-i phi/2), exp(i phi/2)).
    """
    complex_dtype = _complex_dtype(angles.dtype)

    phase_0 = torch.exp(
        -0.5j * angles.to(dtype=complex_dtype)
    )
    phase_1 = torch.exp(
        0.5j * angles.to(dtype=complex_dtype)
    )

    zeros = torch.zeros_like(phase_0)

    row_0 = torch.stack([phase_0, zeros], dim=-1)
    row_1 = torch.stack([zeros, phase_1], dim=-1)

    return torch.stack(
        [row_0, row_1],
        dim=-2,
    )


def _apply_single_qubit_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    wire: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one single-qubit gate to a batched state vector.

    Parameters
    ----------
    state:
        Complex tensor with shape (batch_size, 2**n_qubits).
    gate:
        Either one shared matrix with shape (2, 2), or a batched
        matrix with shape (batch_size, 2, 2).
    wire:
        Target qubit index. Wire 0 is the most significant basis bit.
    n_qubits:
        Total number of qubits.
    """
    if not 0 <= wire < n_qubits:
        raise ValueError(
            f"wire must be in [0, {n_qubits - 1}], received {wire}."
        )

    batch_size = state.shape[0]

    tensor_state = state.reshape(
        batch_size,
        *([2] * n_qubits),
    )

    target_axis = wire + 1

    permutation = [
        axis
        for axis in range(n_qubits + 1)
        if axis != target_axis
    ] + [target_axis]

    inverse_permutation = [
        permutation.index(axis)
        for axis in range(n_qubits + 1)
    ]

    moved_state = tensor_state.permute(permutation)

    if gate.ndim == 2:
        moved_state = torch.einsum(
            "...i,ji->...j",
            moved_state,
            gate,
        )
    elif gate.ndim == 3:
        if gate.shape[0] != batch_size:
            raise ValueError(
                "A batched gate must have the same batch size as the state."
            )

        moved_state = torch.einsum(
            "b...i,bji->b...j",
            moved_state,
            gate,
        )
    else:
        raise ValueError(
            "gate must have shape (2, 2) or (batch_size, 2, 2)."
        )

    return moved_state.permute(
        inverse_permutation
    ).reshape(batch_size, 2**n_qubits)


def _cnot_matrix(
    n_qubits: int,
    control: int,
    target: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the full CNOT unitary for a small state-vector simulator."""
    if control == target:
        raise ValueError("CNOT control and target must be different.")

    dimension = 2**n_qubits
    matrix = torch.zeros(
        (dimension, dimension),
        dtype=dtype,
    )

    control_bit_position = n_qubits - 1 - control
    target_bit_position = n_qubits - 1 - target

    for input_index in range(dimension):
        control_value = (
            input_index >> control_bit_position
        ) & 1

        output_index = input_index

        if control_value == 1:
            output_index ^= 1 << target_bit_position

        matrix[output_index, input_index] = 1.0

    return matrix


def _pauli_z_signs(
    n_qubits: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return Pauli-Z eigenvalue signs for every wire and basis state."""
    dimension = 2**n_qubits
    signs = torch.empty(
        (n_qubits, dimension),
        dtype=dtype,
    )

    for wire in range(n_qubits):
        bit_position = n_qubits - 1 - wire

        for basis_index in range(dimension):
            bit_value = (
                basis_index >> bit_position
            ) & 1

            signs[wire, basis_index] = (
                1.0 if bit_value == 0 else -1.0
            )

    return signs


class FourQubitVariationalLayer(nn.Module):
    """A differentiable four-qubit variational quantum circuit.

    Circuit structure
    -----------------
    1. Start from |0000>.
    2. Encode four classical angles with one RY gate per qubit.
    3. For each variational depth:
       - apply trainable RY and RZ rotations to every qubit;
       - apply the selected CNOT entanglement pattern.
    4. Measure <Z> on every qubit.

    This class performs exact ideal state-vector simulation:
    - no finite shots,
    - no device noise,
    - no quantum hardware.

    All operations are implemented with PyTorch tensors, so gradients and
    higher-order derivatives can flow through both the data-encoding angles
    and the trainable circuit parameters.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        entanglement: EntanglementPattern = "ring",
    ) -> None:
        super().__init__()

        if n_qubits != 4:
            raise ValueError(
                "This controlled WISER model currently requires exactly "
                "four qubits."
            )

        if depth <= 0:
            raise ValueError("depth must be positive.")

        if entanglement not in {
            "none",
            "linear",
            "ring",
        }:
            raise ValueError(
                "entanglement must be 'none', 'linear', or 'ring'."
            )

        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement

        # Two trainable angles per qubit and per depth:
        # index 0 -> RY angle
        # index 1 -> RZ angle
        self.quantum_weights = nn.Parameter(
            torch.empty(
                depth,
                n_qubits,
                2,
            )
        )

        self.reset_parameters()

        # Buffers use float64/complex128 initially. Calling model.to(...)
        # converts them together with the rest of the module.
        complex_dtype = torch.complex128

        for control, target in self._entangling_pairs():
            self.register_buffer(
                f"_cnot_{control}_{target}",
                _cnot_matrix(
                    n_qubits=n_qubits,
                    control=control,
                    target=target,
                    dtype=complex_dtype,
                ),
                persistent=False,
            )

        self.register_buffer(
            "_z_signs",
            _pauli_z_signs(
                n_qubits=n_qubits,
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        """Initialize circuit angles near zero for stable initial training."""
        nn.init.uniform_(
            self.quantum_weights,
            a=-0.10,
            b=0.10,
        )

    def _entangling_pairs(
        self,
    ) -> list[tuple[int, int]]:
        """Return CNOT pairs for the chosen connectivity."""
        if self.entanglement == "none":
            return []

        pairs = [
            (wire, wire + 1)
            for wire in range(self.n_qubits - 1)
        ]

        if self.entanglement == "ring":
            pairs.append(
                (self.n_qubits - 1, 0)
            )

        return pairs

    def _initial_state(
        self,
        batch_size: int,
        real_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a batch of |0000> state vectors."""
        complex_dtype = _complex_dtype(real_dtype)

        state = torch.zeros(
            (batch_size, 2**self.n_qubits),
            dtype=complex_dtype,
            device=device,
        )

        state[:, 0] = 1.0 + 0.0j

        return state

    def forward(
        self,
        encoded_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Return four Pauli-Z expectation values.

        Parameters
        ----------
        encoded_angles:
            Tensor with shape (batch_size, 4).

        Returns
        -------
        torch.Tensor
            Four expectation values with shape (batch_size, 4), each in
            the interval [-1, 1].
        """
        if encoded_angles.ndim != 2:
            raise ValueError(
                "encoded_angles must have shape (batch_size, 4)."
            )

        if encoded_angles.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} encoded angles, "
                f"received {encoded_angles.shape[1]}."
            )

        if not torch.is_floating_point(encoded_angles):
            raise TypeError(
                "encoded_angles must use a floating-point dtype."
            )

        batch_size = encoded_angles.shape[0]

        state = self._initial_state(
            batch_size=batch_size,
            real_dtype=encoded_angles.dtype,
            device=encoded_angles.device,
        )

        # Data encoding: one RY angle per qubit.
        for wire in range(self.n_qubits):
            state = _apply_single_qubit_gate(
                state=state,
                gate=_ry_gate(
                    encoded_angles[:, wire]
                ),
                wire=wire,
                n_qubits=self.n_qubits,
            )

        # Trainable variational circuit.
        for layer_index in range(self.depth):
            for wire in range(self.n_qubits):
                ry_angle = self.quantum_weights[
                    layer_index,
                    wire,
                    0,
                ].expand(batch_size)

                rz_angle = self.quantum_weights[
                    layer_index,
                    wire,
                    1,
                ].expand(batch_size)

                state = _apply_single_qubit_gate(
                    state=state,
                    gate=_ry_gate(ry_angle),
                    wire=wire,
                    n_qubits=self.n_qubits,
                )

                state = _apply_single_qubit_gate(
                    state=state,
                    gate=_rz_gate(rz_angle),
                    wire=wire,
                    n_qubits=self.n_qubits,
                )

            for control, target in self._entangling_pairs():
                cnot = getattr(
                    self,
                    f"_cnot_{control}_{target}",
                ).to(
                    dtype=state.dtype,
                    device=state.device,
                )

                # For row-wise state vectors:
                # |psi_new> = U |psi>  becomes  state @ U^T.
                state = state @ cnot.transpose(0, 1)

        probabilities = torch.abs(state) ** 2

        z_signs = self._z_signs.to(
            dtype=probabilities.dtype,
            device=probabilities.device,
        )

        expectation_values = (
            probabilities @ z_signs.transpose(0, 1)
        )

        # Mathematically, every expectation value lies in [-1, 1].
        # We do not clamp here because clamping could suppress gradients
        # near the physical limits.
        return expectation_values


class HybridQuantumPINN(nn.Module):
    """Quantum-assisted PINN with a four-qubit variational middle layer.

    Default architecture:

        (x, t)
          -> Linear(2, 8) -> tanh
          -> Linear(8, 4) -> tanh          [encoding angles]
          -> four-qubit variational circuit
          -> four <Z> measurements
          -> Linear(4, 8) -> tanh
          -> Linear(8, 1)
          -> u_theta(x, t)
    """

    def __init__(
        self,
        in_features: int = 2,
        classical_width: int = 8,
        n_qubits: int = 4,
        quantum_depth: int = 2,
        entanglement: EntanglementPattern = "ring",
        out_features: int = 1,
    ) -> None:
        super().__init__()

        if in_features != 2:
            raise ValueError(
                "The current PINN expects exactly two inputs: x and t."
            )

        if classical_width <= 0:
            raise ValueError(
                "classical_width must be positive."
            )

        if out_features != 1:
            raise ValueError(
                "The current scalar PDE model requires one output."
            )

        self.in_features = in_features
        self.classical_width = classical_width
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth
        self.entanglement = entanglement
        self.out_features = out_features

        self.input_layer = nn.Linear(
            in_features,
            classical_width,
        )
        self.input_activation = nn.Tanh()

        self.angle_projection = nn.Linear(
            classical_width,
            n_qubits,
        )
        self.angle_activation = nn.Tanh()

        self.quantum_layer = FourQubitVariationalLayer(
            n_qubits=n_qubits,
            depth=quantum_depth,
            entanglement=entanglement,
        )

        self.expansion_layer = nn.Linear(
            n_qubits,
            classical_width,
        )
        self.expansion_activation = nn.Tanh()

        self.output_layer = nn.Linear(
            classical_width,
            out_features,
        )

        self.reset_classical_parameters()

    def reset_classical_parameters(self) -> None:
        """Initialize all classical linear layers."""
        for module in (
            self.input_layer,
            self.angle_projection,
            self.expansion_layer,
            self.output_layer,
        ):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        inputs: torch.Tensor,
    ) -> QuantumPINNOutputs:
        """Return the prediction and all major intermediate representations."""
        self._validate_inputs(inputs)

        classical_features = self.input_activation(
            self.input_layer(inputs)
        )

        # tanh keeps encoding angles in (-1, 1) radians.
        encoded_angles = self.angle_activation(
            self.angle_projection(
                classical_features
            )
        )

        quantum_measurements = self.quantum_layer(
            encoded_angles
        )

        post_quantum_features = (
            self.expansion_activation(
                self.expansion_layer(
                    quantum_measurements
                )
            )
        )

        prediction = self.output_layer(
            post_quantum_features
        )

        return QuantumPINNOutputs(
            classical_features=classical_features,
            encoded_angles=encoded_angles,
            quantum_measurements=quantum_measurements,
            post_quantum_features=post_quantum_features,
            prediction=prediction,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar PINN prediction u_theta(x, t)."""
        return self.encode(inputs).prediction

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
    ) -> None:
        """Validate the PINN input tensor."""
        if inputs.ndim != 2:
            raise ValueError(
                "inputs must have shape "
                "(number_of_points, 2)."
            )

        if inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} features, "
                f"received {inputs.shape[1]}."
            )

        if not torch.is_floating_point(inputs):
            raise TypeError(
                "inputs must use a floating-point dtype."
            )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def quantum_parameter_count(
    model: HybridQuantumPINN,
) -> int:
    """Return only the number of trainable circuit parameters."""
    return model.quantum_layer.quantum_weights.numel()


def _finite_difference_checks(
    model: nn.Module,
    point: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Compare autograd x-derivatives with central finite differences."""
    point = point.clone().detach().requires_grad_(True)

    prediction = model(point)

    first_derivatives = torch.autograd.grad(
        outputs=prediction,
        inputs=point,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = first_derivatives[:, 0:1]

    second_derivatives = torch.autograd.grad(
        outputs=u_x,
        inputs=point,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = second_derivatives[:, 0:1]

    step_first = 1.0e-5
    step_second = 1.0e-3

    point_plus_first = point.detach().clone()
    point_minus_first = point.detach().clone()

    point_plus_first[:, 0] += step_first
    point_minus_first[:, 0] -= step_first

    finite_u_x = (
        model(point_plus_first)
        - model(point_minus_first)
    ) / (2.0 * step_first)

    point_plus_second = point.detach().clone()
    point_minus_second = point.detach().clone()

    point_plus_second[:, 0] += step_second
    point_minus_second[:, 0] -= step_second

    finite_u_xx = (
        model(point_plus_second)
        - 2.0 * model(point.detach())
        + model(point_minus_second)
    ) / (step_second**2)

    first_error = float(
        torch.max(
            torch.abs(u_x.detach() - finite_u_x)
        )
    )

    second_error = float(
        torch.max(
            torch.abs(u_xx.detach() - finite_u_xx)
        )
    )

    return (
        float(u_x.item()),
        float(finite_u_x.item()),
        first_error,
        second_error,
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    model = HybridQuantumPINN(
        in_features=2,
        classical_width=8,
        n_qubits=4,
        quantum_depth=2,
        entanglement="ring",
        out_features=1,
    ).to(dtype=torch.float64)

    inputs = torch.tensor(
        [
            [0.20, 0.10],
            [0.50, 0.50],
            [0.80, 0.90],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    outputs = model.encode(inputs)

    assert outputs.classical_features.shape == (3, 8)
    assert outputs.encoded_angles.shape == (3, 4)
    assert outputs.quantum_measurements.shape == (3, 4)
    assert outputs.post_quantum_features.shape == (3, 8)
    assert outputs.prediction.shape == (3, 1)

    tolerance = 1.0e-10
    assert torch.all(
        outputs.quantum_measurements <= 1.0 + tolerance
    )
    assert torch.all(
        outputs.quantum_measurements >= -1.0 - tolerance
    )

    parameter_count = count_trainable_parameters(
        model
    )
    circuit_parameter_count = (
        quantum_parameter_count(model)
    )

    # Classical parameters:
    # 2->8: 24
    # 8->4: 36
    # 4->8: 40
    # 8->1: 9
    # Quantum circuit:
    # depth * qubits * (RY,RZ) = 2*4*2 = 16
    # Total = 24+36+40+9+16 = 125
    assert parameter_count == 125
    assert circuit_parameter_count == 16

    prediction = model(inputs)

    first_derivatives = torch.autograd.grad(
        outputs=prediction,
        inputs=inputs,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = first_derivatives[:, 0:1]
    u_t = first_derivatives[:, 1:2]

    second_derivatives = torch.autograd.grad(
        outputs=u_x,
        inputs=inputs,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = second_derivatives[:, 0:1]

    assert torch.all(torch.isfinite(u_x))
    assert torch.all(torch.isfinite(u_t))
    assert torch.all(torch.isfinite(u_xx))

    # Verify that gradients reach trainable quantum angles.
    loss = torch.mean(prediction**2)
    quantum_gradient = torch.autograd.grad(
        outputs=loss,
        inputs=model.quantum_layer.quantum_weights,
        retain_graph=True,
    )[0]

    assert torch.all(torch.isfinite(quantum_gradient))
    assert torch.linalg.vector_norm(
        quantum_gradient
    ).item() > 0.0

    (
        autograd_u_x,
        finite_u_x,
        first_derivative_error,
        second_derivative_error,
    ) = _finite_difference_checks(
        model=model,
        point=torch.tensor(
            [[0.37, 0.41]],
            dtype=torch.float64,
        ),
    )

    assert first_derivative_error < 1.0e-7
    assert second_derivative_error < 5.0e-5

    print(
        "Four-qubit hybrid quantum PINN "
        "structural self-check passed."
    )
    print(
        "Architecture: "
        "2 -> 8 -> 4 encoding angles -> "
        "4-qubit VQC -> 4 measurements -> 8 -> 1"
    )
    print(
        f"Entanglement pattern: {model.entanglement}"
    )
    print(
        f"Variational depth: {model.quantum_depth}"
    )
    print(
        f"Total trainable parameters: {parameter_count}"
    )
    print(
        "Trainable quantum-circuit parameters: "
        f"{circuit_parameter_count}"
    )
    print(
        "Quantum measurement range observed: "
        f"[{outputs.quantum_measurements.min().item():.6f}, "
        f"{outputs.quantum_measurements.max().item():.6f}]"
    )
    print(
        "Quantum-gradient norm: "
        f"{torch.linalg.vector_norm(quantum_gradient).item():.6e}"
    )
    print(
        "Autograd u_x vs finite-difference u_x: "
        f"{autograd_u_x:.8f} vs {finite_u_x:.8f}"
    )
    print(
        "First-derivative absolute error: "
        f"{first_derivative_error:.3e}"
    )
    print(
        "Second-derivative absolute error: "
        f"{second_derivative_error:.3e}"
    )
    print(
        "u_x, u_t, and u_xx checks passed."
    )
