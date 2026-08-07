from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BottleneckOutputs:
    """Intermediate representations produced by the bottleneck PINN."""

    first_hidden: torch.Tensor
    bottleneck: torch.Tensor
    second_hidden: torch.Tensor
    prediction: torch.Tensor


class ClassicalBottleneckPINN(nn.Module):
    """Classical PINN with a controlled low-dimensional bottleneck.

    Default architecture:

        (x, t) -> 8 -> 4 -> 8 -> u_theta(x, t)

    The four-dimensional middle layer is designed as a fair classical
    control for a future four-qubit quantum layer.
    """

    def __init__(
        self,
        in_features: int = 2,
        classical_width: int = 8,
        bottleneck_dimension: int = 4,
        out_features: int = 1,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if classical_width <= 0:
            raise ValueError("classical_width must be positive.")
        if bottleneck_dimension <= 0:
            raise ValueError(
                "bottleneck_dimension must be positive."
            )
        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        self.in_features = in_features
        self.classical_width = classical_width
        self.bottleneck_dimension = bottleneck_dimension
        self.out_features = out_features

        self.input_layer = nn.Linear(
            in_features,
            classical_width,
        )
        self.input_activation = nn.Tanh()

        self.bottleneck_layer = nn.Linear(
            classical_width,
            bottleneck_dimension,
        )
        self.bottleneck_activation = nn.Tanh()

        self.expansion_layer = nn.Linear(
            bottleneck_dimension,
            classical_width,
        )
        self.expansion_activation = nn.Tanh()

        self.output_layer = nn.Linear(
            classical_width,
            out_features,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear weights with Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, inputs: torch.Tensor) -> BottleneckOutputs:
        """Return the prediction together with hidden representations."""
        self._validate_inputs(inputs)

        first_hidden = self.input_activation(
            self.input_layer(inputs)
        )

        bottleneck = self.bottleneck_activation(
            self.bottleneck_layer(first_hidden)
        )

        second_hidden = self.expansion_activation(
            self.expansion_layer(bottleneck)
        )

        prediction = self.output_layer(second_hidden)

        return BottleneckOutputs(
            first_hidden=first_hidden,
            bottleneck=bottleneck,
            second_hidden=second_hidden,
            prediction=prediction,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the scalar PINN prediction u_theta(x, t)."""
        return self.encode(inputs).prediction

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate model input shape and dtype."""
        if inputs.ndim != 2:
            raise ValueError(
                "inputs must have shape "
                "(number_of_points, number_of_features)."
            )

        if inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"but received {inputs.shape[1]}."
            )

        if not torch.is_floating_point(inputs):
            raise TypeError(
                "inputs must use a floating-point dtype."
            )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ClassicalBottleneckPINN(
        in_features=2,
        classical_width=8,
        bottleneck_dimension=4,
        out_features=1,
    ).to(dtype=torch.float64)

    inputs = torch.tensor(
        [
            [0.25, 0.10],
            [0.50, 0.50],
            [0.75, 0.90],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    outputs = model.encode(inputs)

    assert outputs.first_hidden.shape == (3, 8)
    assert outputs.bottleneck.shape == (3, 4)
    assert outputs.second_hidden.shape == (3, 8)
    assert outputs.prediction.shape == (3, 1)

    assert torch.all(outputs.bottleneck < 1.0)
    assert torch.all(outputs.bottleneck > -1.0)

    parameter_count = count_trainable_parameters(model)
    assert parameter_count == 109

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

    print("Classical bottleneck structural self-check passed.")
    print("Architecture: 2 -> 8 -> 4 -> 8 -> 1")
    print(f"Trainable parameters: {parameter_count}")
    print("Bottleneck output shape:", outputs.bottleneck.shape)
    print(
        "Observed bottleneck range: "
        f"[{outputs.bottleneck.min().item():.6f}, "
        f"{outputs.bottleneck.max().item():.6f}]"
    )
    print("Second-derivative check passed.")
