from torch import nn
from vicentin.deep_learning.loss import BaseLoss


class WrapTorchLoss(BaseLoss):
    def __init__(self, torch_loss: nn.Module):
        super().__init__()
        self.criterion = torch_loss

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss, {}
