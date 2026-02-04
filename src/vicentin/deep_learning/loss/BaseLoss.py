import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        pass
