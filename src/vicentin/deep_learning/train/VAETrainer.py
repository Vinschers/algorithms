import os
from typing import Dict, Any, Optional, Literal

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from vicentin.deep_learning.loss import BaseLoss
from vicentin.deep_learning.train import GenericTrainer


class VAETrainer(GenericTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        hyperparams: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        float16: bool = False,
        sample_path: str = "vae_samples",
        n_imgs_val: int = 8,
        export_mode: Literal["save", "show", None] = "show",
    ):
        super().__init__(
            models={"vae": model},
            optimizers={"opt": optimizer},
            hyperparams=hyperparams,
            device=device,
            float16=float16,
        )
        self.loss_fn = loss_fn
        self.sample_path = sample_path
        self.n_imgs_val = n_imgs_val
        self.export_mode = export_mode

        if self.export_mode == "save" and self.n_imgs_val > 0:
            os.makedirs(self.sample_path, exist_ok=True)

    def train_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            real_img = batch[0]
        else:
            real_img = batch

        results = self.models["vae"](real_img)
        loss, log_metrics = self.loss_fn(results, real_img)
        self.optimize(loss, self.optimizers["opt"])

        return {**{"loss": loss.item()}, **log_metrics}

    def validate_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            real_img = batch[0]
        else:
            real_img = batch

        results = self.models["vae"](real_img)
        loss, log_metrics = self.loss_fn(results, real_img)

        if (
            batch_idx == 0
            and self.n_imgs_val > 0
            and self.export_mode != "none"
        ):
            self._handle_visualization(real_img, results)

        return {**{"loss": loss.item()}, **log_metrics}

    def _handle_visualization(self, real, results):
        recon = results[0]

        n = min(real.size(0), self.n_imgs_val)
        real_slice = real[:n]
        recon_slice = recon[:n]

        comparison = torch.cat([real_slice, recon_slice])
        grid = make_grid(comparison, nrow=n, normalize=True)

        if self.export_mode == "save":
            path = os.path.join(
                self.sample_path, f"epoch_{self.current_epoch}_recon.png"
            )
            save_image(grid, path)

        elif self.export_mode == "show":
            img = (
                grid.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            plt.figure(figsize=(n * 2, 4))
            plt.imshow(img)
            plt.title(
                f"Reconstruction (Top: Real, Bottom: Recon) - Epoch {self.current_epoch}"
            )
            plt.axis("off")
            plt.show()
