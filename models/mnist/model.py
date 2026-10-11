"""Small MNIST classifier. The module returns logits and does not train itself."""

from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 784
HIDDEN_SIZE = 128
SECOND_HIDDEN_SIZE = 64
CLASS_COUNT = 10
PARAMETER_COUNT = 109_386


class MnistMlp(nn.Module):
    """Linear(784, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, SECOND_HIDDEN_SIZE)
        self.fc3 = nn.Linear(SECOND_HIDDEN_SIZE, CLASS_COUNT)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(images))
        hidden = torch.relu(self.fc2(hidden))
        return self.fc3(hidden)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
