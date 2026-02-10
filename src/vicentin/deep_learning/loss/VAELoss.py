import torch
import torch.nn as nn
from typing import Optional

from vicentin.deep_learning.loss import (
    BaseLoss,
    GaussianKLDivergenceLoss,
    WrapTorchLoss,
)


class VAELoss(BaseLoss):
    def __init__(
        self, reconstruction_loss: Optional[nn.Module] = None, beta: float = 1.0
    ):
        super().__init__()

        if reconstruction_loss is None:
            reconstruction_loss = nn.MSELoss()

        if not isinstance(reconstruction_loss, BaseLoss):
            reconstruction_loss = WrapTorchLoss(reconstruction_loss)

        self.reconstruction_loss = reconstruction_loss
        self.beta = beta
        self.kl_loss = GaussianKLDivergenceLoss()

    def forward(self, output, x) -> tuple[torch.Tensor, dict]:
        x_hat, mu, logvar = output

        reconstruction_loss, reconstruction_metrics = self.reconstruction_loss(
            x_hat, x
        )
        kl_loss, kl_metrics = self.kl_loss(mu, logvar)

        loss = reconstruction_loss + self.beta * kl_loss

        metrics = {
            "reconstruction": reconstruction_loss.item(),
            "KL": kl_loss.item(),
            "loss": loss.item(),
        }

        for name, value in reconstruction_metrics.items():
            metrics[f"reconstruction_{name}"] = value

        for name, value in kl_metrics.items():
            metrics[f"KL_{name}"] = value

        return loss, metrics
