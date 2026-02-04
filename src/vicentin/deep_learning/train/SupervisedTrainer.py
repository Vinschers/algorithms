from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from vicentin.deep_learning.loss import BaseLoss
from vicentin.deep_learning.train import StandardTrainer


class SupervisedTrainer(StandardTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        hyperparams: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        float16: bool = False,
    ):
        super().__init__(
            models={"model": model},
            optimizers={"opt": optimizer},
            hyperparams=hyperparams,
            device=device,
            float16=float16,
        )
        self.loss_fn = loss_fn

    def compute_loss(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            raise ValueError(
                "SupervisedTrainer expects batch to be a tuple of (input, target)"
            )

        y_pred = self.models["model"](x)

        loss, metrics = self.loss_fn(y_pred, y)

        if "loss" not in metrics:
            metrics["loss"] = loss.item()

        return loss, metrics
