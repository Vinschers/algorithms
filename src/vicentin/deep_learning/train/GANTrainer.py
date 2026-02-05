import os
from typing import Dict, Any, Optional, Literal

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from vicentin.deep_learning.loss import BaseLoss
from vicentin.deep_learning.train import GenericTrainer


class GANTrainer(GenericTrainer):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        G_opt: torch.optim.Optimizer,
        D_opt: torch.optim.Optimizer,
        G_loss: BaseLoss,
        D_loss: BaseLoss,
        D_steps: int = 1,
        noise_dim: int = 100,
        n_imgs_val: int = 64,
        sample_path: str = "gan_samples",
        export_mode: Literal["save", "show", None] = None,
        fixed_z: Optional[torch.Tensor] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        float16: bool = False,
    ):
        super().__init__(
            models={"G": generator, "D": discriminator},
            optimizers={"G_opt": G_opt, "D_opt": D_opt},
            hyperparams=hyperparams,
            device=device,
            float16=float16,
        )

        self.D_loss = D_loss
        self.G_loss = G_loss
        self.D_steps = D_steps
        self.noise_dim = noise_dim

        self.n_imgs_val = n_imgs_val
        self.sample_path = sample_path
        self.export_mode = export_mode

        if self.export_mode == "save" and self.n_imgs_val > 0:
            os.makedirs(self.sample_path, exist_ok=True)

        self.fixed_z = fixed_z
        if self.fixed_z is None and self.n_imgs_val > 0:
            self.fixed_z = torch.randn(
                self.n_imgs_val, noise_dim, device=self.device
            )

    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        if isinstance(batch, (list, tuple)):
            real_imgs = batch[0]
        else:
            real_imgs = batch

        z = torch.randn(real_imgs.size(0), self.noise_dim, device=self.device)
        fake_imgs = self.models["G"](z).detach()

        d_loss, d_metrics = self.D_loss(real_imgs, fake_imgs)

        self.optimize(d_loss, self.optimizers["D_opt"])

        g_metrics = {}

        if batch_idx % self.D_steps == 0:
            fake_imgs_g = self.models["G"](z)
            g_loss, g_metrics = self.G_loss(fake_imgs_g)
            self.optimize(g_loss, self.optimizers["G_opt"])

        return {**d_metrics, **g_metrics}

    def validate_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        if (
            batch_idx == 0
            and self.fixed_z is not None
            and self.export_mode is not None
        ):
            self._handle_visualization()

        if isinstance(batch, (list, tuple)):
            real_imgs = batch[0]
        else:
            real_imgs = batch

        z = torch.randn(real_imgs.size(0), self.noise_dim, device=self.device)
        fake_imgs = self.models["G"](z)

        _, d_metrics = self.D_loss(real_imgs, fake_imgs)
        _, g_metrics = self.G_loss(fake_imgs)

        return {**d_metrics, **g_metrics}

    def _handle_visualization(self):
        self.models["G"].eval()

        with torch.no_grad():
            fake_grid = self.models["G"](self.fixed_z)

            grid = make_grid(fake_grid, normalize=True)

            if self.export_mode == "save":
                path = os.path.join(
                    self.sample_path, f"epoch_{self.current_epoch}.png"
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
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"Generated Samples - Epoch {self.current_epoch}")
                plt.axis("off")
                plt.show()
