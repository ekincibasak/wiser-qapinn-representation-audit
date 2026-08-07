from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


TrigonometricConnectivity = Literal["local", "ring"]


@dataclass
class TrigonometricPINNOutputs:
    """Intermediate values produced by the classical trigonometric PINN."""

    classical_features: torch.Tensor
    encoded_angles: torch.Tensor
    trigonometric_features: torch.Tensor
    post_feature_map: torch.Tensor
    prediction: torch.Tensor


class FixedTrigonometricFeatureMap(nn.Module):
    """A fixed classical trigonometric control for the frozen quantum layer.

    The input is a four-dimensional angle vector. The map contains sixteen
    fixed phase values with shape (depth=2, features=4, rotations=2), matching
    the number and initialization range of the frozen four-qubit circuit's
    fixed variational angles.

    Two controls are supported:

    local:
        Each output uses only its own encoded angle.

    ring:
        Each output multiplies local cosine features with those of its next
        neighbor on a four-node ring.

    This is not a quantum simulation. It is an explicit classical nonlinear
    feature map built from ordinary trigonometric operations.
    """

    def __init__(
        self,
        dimension: int = 4,
        depth: int = 2,
        connectivity: TrigonometricConnectivity = "ring",
    ) -> None:
        super().__init__()

        if dimension != 4:
            raise ValueError(
                "This controlled comparison currently requires dimension=4."
            )
        if depth != 2:
            raise ValueError(
                "This controlled comparison currently requires depth=2."
            )
        if connectivity not in {"local", "ring"}:
            raise ValueError(
                "connectivity must be 'local' or 'ring'."
            )

        self.dimension = dimension
        self.depth = depth
        self.connectivity = connectivity

        fixed_phases = torch.empty(
            depth,
            dimension,
            2,
        )
        nn.init.uniform_(
            fixed_phases,
            a=-0.10,
            b=0.10,
        )

        self.register_buffer(
            "fixed_phases",
            fixed_phases,
            persistent=True,
        )

    def forward(
        self,
        encoded_angles: torch.Tensor,
    ) -> torch.Tensor:
        """Return four bounded classical trigonometric features."""
        if encoded_angles.ndim != 2:
            raise ValueError(
                "encoded_angles must have shape (batch_size, 4)."
            )
        if encoded_angles.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} angles, "
                f"received {encoded_angles.shape[1]}."
            )
        if not torch.is_floating_point(encoded_angles):
            raise TypeError(
                "encoded_angles must use a floating-point dtype."
            )

        phases = self.fixed_phases.to(
            dtype=encoded_angles.dtype,
            device=encoded_angles.device,
        )

        # Four local cosine responses for each of two depth blocks.
        first_a = torch.cos(
            encoded_angles + phases[0, :, 0]
        )
        first_b = torch.cos(
            encoded_angles + phases[0, :, 1]
        )
        second_a = torch.cos(
            encoded_angles + phases[1, :, 0]
        )
        second_b = torch.cos(
            encoded_angles + phases[1, :, 1]
        )

        if self.connectivity == "local":
            features = 0.5 * (
                first_a * first_b
                + second_a * second_b
            )
        else:
            # A minimal classical analogue of ring-coupled Z expectations:
            # each feature combines one local factor and one next-neighbor
            # factor. The product stays inside [-1, 1].
            first_neighbor = torch.roll(
                first_b,
                shifts=-1,
                dims=1,
            )
            second_neighbor = torch.roll(
                second_b,
                shifts=-1,
                dims=1,
            )

            features = 0.5 * (
                first_a * first_neighbor
                + second_a * second_neighbor
            )

        return features


class ClassicalTrigonometricPINN(nn.Module):
    """PINN with a fixed classical trigonometric four-value feature map.

    Architecture:

        (x,t)
          -> Linear(2,8) -> tanh
          -> Linear(8,4) -> tanh
          -> fixed trigonometric feature map
          -> Linear(4,8) -> tanh
          -> Linear(8,1)

    Trainable parameter count: 109.
    Fixed phase constants: 16.
    """

    def __init__(
        self,
        in_features: int = 2,
        classical_width: int = 8,
        feature_dimension: int = 4,
        feature_depth: int = 2,
        connectivity: TrigonometricConnectivity = "ring",
        out_features: int = 1,
    ) -> None:
        super().__init__()

        if in_features != 2:
            raise ValueError(
                "The current PDE model expects inputs (x,t)."
            )
        if classical_width <= 0:
            raise ValueError(
                "classical_width must be positive."
            )
        if feature_dimension != 4:
            raise ValueError(
                "feature_dimension must equal 4."
            )
        if out_features != 1:
            raise ValueError(
                "The current scalar PDE model requires one output."
            )

        self.in_features = in_features
        self.classical_width = classical_width
        self.feature_dimension = feature_dimension
        self.feature_depth = feature_depth
        self.connectivity = connectivity
        self.out_features = out_features

        # The construction order intentionally matches HybridQuantumPINN:
        # input layer -> angle projection -> fixed middle layer ->
        # expansion -> output -> classical Xavier reset.
        self.input_layer = nn.Linear(
            in_features,
            classical_width,
        )
        self.input_activation = nn.Tanh()

        self.angle_projection = nn.Linear(
            classical_width,
            feature_dimension,
        )
        self.angle_activation = nn.Tanh()

        self.feature_map = FixedTrigonometricFeatureMap(
            dimension=feature_dimension,
            depth=feature_depth,
            connectivity=connectivity,
        )

        self.expansion_layer = nn.Linear(
            feature_dimension,
            classical_width,
        )
        self.expansion_activation = nn.Tanh()

        self.output_layer = nn.Linear(
            classical_width,
            out_features,
        )

        self.reset_classical_parameters()

    def reset_classical_parameters(self) -> None:
        """Use the same classical initialization as the hybrid model."""
        for module in (
            self.input_layer,
            self.angle_projection,
            self.expansion_layer,
            self.output_layer,
        ):
            nn.init.xavier_normal_(
                module.weight
            )
            nn.init.zeros_(
                module.bias
            )

    def encode(
        self,
        inputs: torch.Tensor,
    ) -> TrigonometricPINNOutputs:
        """Return predictions and internal representations."""
        self._validate_inputs(inputs)

        classical_features = self.input_activation(
            self.input_layer(inputs)
        )
        encoded_angles = self.angle_activation(
            self.angle_projection(
                classical_features
            )
        )
        trigonometric_features = self.feature_map(
            encoded_angles
        )
        post_feature_map = (
            self.expansion_activation(
                self.expansion_layer(
                    trigonometric_features
                )
            )
        )
        prediction = self.output_layer(
            post_feature_map
        )

        return TrigonometricPINNOutputs(
            classical_features=classical_features,
            encoded_angles=encoded_angles,
            trigonometric_features=trigonometric_features,
            post_feature_map=post_feature_map,
            prediction=prediction,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar PDE prediction."""
        return self.encode(inputs).prediction

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
    ) -> None:
        if inputs.ndim != 2:
            raise ValueError(
                "inputs must have shape (number_of_points, 2)."
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
    """Count trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_fixed_phase_values(
    model: ClassicalTrigonometricPINN,
) -> int:
    """Count fixed phase constants in the feature map."""
    return model.feature_map.fixed_phases.numel()


if __name__ == "__main__":
    torch.manual_seed(42)

    for connectivity in ("local", "ring"):
        model = ClassicalTrigonometricPINN(
            connectivity=connectivity,
        ).to(dtype=torch.float64)

        inputs = torch.tensor(
            [
                [-0.75, 0.10],
                [0.00, 0.50],
                [0.80, 0.90],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        outputs = model.encode(inputs)

        assert outputs.encoded_angles.shape == (3, 4)
        assert outputs.trigonometric_features.shape == (3, 4)
        assert outputs.prediction.shape == (3, 1)

        tolerance = 1.0e-12
        assert torch.all(
            outputs.trigonometric_features
            <= 1.0 + tolerance
        )
        assert torch.all(
            outputs.trigonometric_features
            >= -1.0 - tolerance
        )

        assert count_trainable_parameters(model) == 109
        assert count_fixed_phase_values(model) == 16

        prediction = model(inputs)
        first = torch.autograd.grad(
            outputs=prediction,
            inputs=inputs,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = first[:, 0:1]
        u_t = first[:, 1:2]

        second = torch.autograd.grad(
            outputs=u_x,
            inputs=inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_xx = second[:, 0:1]

        assert torch.all(torch.isfinite(u_x))
        assert torch.all(torch.isfinite(u_t))
        assert torch.all(torch.isfinite(u_xx))

        print(
            "Classical trigonometric PINN self-check passed: "
            f"{connectivity}."
        )
        print(
            "Architecture: "
            "2 -> 8 -> 4 angles -> fixed trig map -> 4 -> 8 -> 1"
        )
        print(
            "Trainable parameters: "
            f"{count_trainable_parameters(model)}"
        )
        print(
            "Fixed phase values: "
            f"{count_fixed_phase_values(model)}"
        )
        print(
            "Observed feature range: "
            f"[{outputs.trigonometric_features.min().item():.6f}, "
            f"{outputs.trigonometric_features.max().item():.6f}]"
        )
        print(
            "u_x, u_t, and u_xx checks passed."
        )
