from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vicentin.deep_learning.loss import BaseLoss


class WeightedSumLoss(BaseLoss):
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Computes a weighted sum of multiple loss functions.

        Args:
            losses: Dictionary of {name: loss_module}.
                    Example: {'l1': nn.L1Loss(), 'perceptual': VGGPerceptualLoss()}
            weights: Optional dictionary of {name: weight}. Defaults to 1.0 for all.
        """
        super().__init__()

        self.losses = nn.ModuleDict(losses)

        if weights is None:
            self.weights = {k: 1.0 for k in losses.keys()}
        else:
            assert (
                weights.keys() == losses.keys()
            ), "Weights keys must match losses keys."
            self.weights = weights

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        all_metrics = {}

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]

            if weight == 0:
                continue

            if isinstance(loss_fn, BaseLoss):
                loss, sub_metrics = loss_fn(*args, **kwargs)

                for k, v in sub_metrics.items():
                    all_metrics[f"{name}_{k}"] = v
            else:
                loss = loss_fn(*args, **kwargs)

            total_loss = total_loss + (loss * weight)

            all_metrics[name] = loss.item()

        if isinstance(total_loss, float):
            return torch.tensor(total_loss, requires_grad=True), all_metrics
        elif isinstance(total_loss, torch.Tensor):
            return total_loss, all_metrics
        else:
            raise RuntimeError("Loss must be a float or torch.Tensor.")
