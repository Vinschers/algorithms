import torch
from torch import nn

from vicentin.deep_learning.loss import BaseLoss


class WassersteinDiscriminatorLoss(BaseLoss):
    def __init__(self, D: nn.Module, lambda_gp: float = 10.0):
        super().__init__()
        self.D = D
        self.lambda_gp = lambda_gp

    def compute_gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        device = real.device

        alpha_shape = [batch_size] + [1] * (real.ndim - 1)
        alpha = torch.rand(alpha_shape, device=device)

        interpolates = alpha * real + ((1 - alpha) * fake)
        interpolates.requires_grad_(True)

        d_interpolates = self.D(interpolates)
        fake_grad = torch.ones(d_interpolates.size(), device=device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gp

    def forward(self, real_imgs, fake_imgs):
        real_loss = self.D(real_imgs)
        fake_loss = self.D(fake_imgs)

        gp = self.compute_gradient_penalty(real_imgs, fake_imgs)

        loss = torch.mean(fake_loss - real_loss) + self.lambda_gp * gp

        w_dist = torch.mean(real_loss) - torch.mean(fake_loss)

        return loss, {
            "real_loss": real_loss.mean().item(),
            "fake_loss": fake_loss.mean().item(),
            "W_dist": w_dist.item(),
            "GP": gp.item(),
        }


class WassersteinGeneratorLoss(BaseLoss):
    def __init__(self, D: nn.Module):
        super().__init__()
        self.D = D

    def forward(self, fake_imgs):
        fake_validity = self.D(fake_imgs)
        loss = -torch.mean(fake_validity)

        return loss, {"G_loss": loss.item()}
