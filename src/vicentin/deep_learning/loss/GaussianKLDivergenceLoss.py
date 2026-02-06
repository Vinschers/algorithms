import torch
from typing import Any, Optional, Tuple, Dict

from vicentin.deep_learning.loss import BaseLoss


class GaussianKLDivergenceLoss(BaseLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: Optional[torch.Tensor] = None,
        logvar2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        var1 = logvar1.exp()
        var2 = None

        if mu2 is None:
            kld_element = -0.5 * (1 + logvar1 - mu1.pow(2) - var1)
        else:
            if logvar2 is None:
                logvar2 = torch.zeros_like(mu2)

            var2 = logvar2.exp()

            numerator = var1 + (mu1 - mu2).pow(2)
            kld_element = 0.5 * ((numerator / var2) - 1 + logvar2 - logvar1)

        kld = torch.sum(kld_element.flatten(1), dim=1)

        if self.reduction == "mean":
            loss = torch.mean(kld)
        elif self.reduction == "sum":
            loss = torch.sum(kld)
        else:
            loss = kld

        metrics = {
            "mean_1_avg": mu1.mean().item(),
            "var_1_avg": var1.mean().item(),
            "var_1_min": var1.min().item(),
            "var_1_max": var1.max().item(),
        }

        if mu2 is not None:
            metrics["mean_2_avg"] = mu2.mean().item()
            if var2 is not None:
                metrics["var_2_avg"] = var2.mean().item()

        return loss, metrics
