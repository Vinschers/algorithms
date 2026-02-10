from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn

from vicentin.deep_learning.loss import SupInfoNCELoss
from vicentin.deep_learning.train import StandardTrainer


class SimCLRTrainer(StandardTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
        float16: bool = True,
    ):
        models = {"SimCLR": model}
        optimizers = {"SimCLR": optimizer}

        if scheduler is not None:
            schedulers = {"SimCLR": scheduler}
        else:
            schedulers = None

        hyperparams = {"SimCLR": temperature}

        self.loss_fn = SupInfoNCELoss(temperature, 0)

        super().__init__(
            models, optimizers, schedulers, hyperparams, device, float16
        )

    def compute_loss(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not isinstance(batch, (list, tuple)):
            raise RuntimeError("Batch should have 2 views.")

        if isinstance(batch[0], (list, tuple)):
            (x1, x2), _ = batch
        else:
            x1, x2 = batch

        x = torch.cat([x1, x2], dim=0)

        model = self.models["SimCLR"]
        _, z = model(x)

        z1, z2 = z.chunk(2, dim=0)

        return self.loss_fn(z1, z2)
