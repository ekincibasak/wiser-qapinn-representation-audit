from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ClassicalMLP(nn.Module):
    """A classical multilayer perceptron for representing u(x, t)."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_layers: Sequence[int] = (16, 16),
    ) -> None:
        """Initialize the multilayer perceptron.

        Parameters
        ----------
        in_features:
            Number of input variables. For our PINN, the inputs are x and t,
            so the default value is 2.
        out_features:
            Number of model outputs. The model predicts u(x, t), so the
            default value is 1.
        hidden_layers:
            Number of neurons in each hidden layer.
        """
        super().__init__()

        dimensions = [
            in_features,
            *hidden_layers,
            out_features,
        ]

        layers: list[nn.Module] = []

        for input_dimension, output_dimension in zip(
            dimensions[:-2],
            dimensions[1:-1],
        ):
            layers.append(
                nn.Linear(
                    input_dimension,
                    output_dimension,
                )
            )
            layers.append(nn.Tanh())

        layers.append(
            nn.Linear(
                dimensions[-2],
                dimensions[-1],
            )
        )

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all linear-layer weights and biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted solution u_theta(x, t)."""
        if inputs.ndim != 2:
            raise ValueError(
                "inputs must have shape (number_of_points, number_of_features)"
            )

        if inputs.shape[1] != 2:
            raise ValueError(
                "The default PINN expects two input features: x and t."
            )

        return self.network(inputs)


if __name__ == "__main__":
    # Create the default model:
    # (x, t) -> 16 neurons -> 16 neurons -> u_theta(x, t)
    model = ClassicalMLP()

    # Example physical coordinates.
    x = torch.tensor([[0.5]], dtype=torch.float32)
    t = torch.tensor([[0.2]], dtype=torch.float32)

    # Combine x and t into one input tensor with shape (1, 2).
    inputs = torch.cat([x, t], dim=1)

    # The output is random at this stage because the model is not trained.
    prediction = model(inputs)

    print(model)
    print("Input shape:", inputs.shape)
    print("Output shape:", prediction.shape)
    print("Predicted u_theta(x, t):", prediction)
