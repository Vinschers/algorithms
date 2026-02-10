from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn

from vicentin.deep_learning.loss import SupInfoNCELoss
from vicentin.deep_learning.train import StandardTrainer


class SimCLRTrainer(StandardTrainer):
    """
    Trainer specific to SimCLR contrastive learning.

    This trainer handles the unpacking of multi-view batches, the calculation of
    InfoNCE loss, and automatically manages Batch Normalization synchronization
    in distributed settings.

    Attributes
    ----------
    loss_fn : SupInfoNCELoss
        The contrastive loss function (InfoNCE).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
        float16: bool = True,
    ) -> None:
        """
        Initialize the SimCLR Trainer.

        Parameters
        ----------
        model : nn.Module
            The SimCLR model to train.
        optimizer : torch.optim.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            Learning rate scheduler.
        temperature : float, optional
            Temperature parameter (tau) for the InfoNCE loss. Default is 0.07.
        device : torch.device, optional
            Device to train on.
        float16 : bool, optional
            Whether to use mixed precision training (AMP). Default is True.
        """
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            print(
                "SimCLRTrainer: Distributed environment detected. Converting to SyncBatchNorm."
            )
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        models = {"SimCLR": model}
        optimizers = {"SimCLR": optimizer}

        if scheduler is not None:
            schedulers = {"SimCLR": scheduler}
        else:
            schedulers = None

        hyperparams = {"SimCLR_tau": temperature}

        self.loss_fn = SupInfoNCELoss(temperature, epsilon=0)

        super().__init__(
            models, optimizers, schedulers, hyperparams, device, float16
        )

    def compute_loss(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the InfoNCE loss for a batch of multi-view images.

        Parameters
        ----------
        batch : Any
            The input batch. Expected to be a list/tuple of 2 tensors [view1, view2],
            or a tuple of ((view1, view2), labels).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            The calculated loss and a dictionary of metrics for logging.

        Raises
        ------
        RuntimeError
            If the batch structure does not contain exactly two views.
        """

        if isinstance(batch, (list, tuple)) and isinstance(
            batch[0], (list, tuple)
        ):
            views, _ = batch
        elif isinstance(batch, (list, tuple)):
            views = batch
        else:
            raise RuntimeError(
                f"Batch format not recognized. Expected list of views, got {type(batch)}."
            )

        if len(views) != 2:
            raise RuntimeError(
                f"SimCLR requires exactly 2 views, got {len(views)}."
            )

        x1, x2 = views

        x = torch.cat([x1, x2], dim=0)

        model = self.models["SimCLR"]

        _, z = model(x)
        z1, z2 = z.chunk(2, dim=0)

        loss, metrics = self.loss_fn(z1, z2)

        return loss, metrics
