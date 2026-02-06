from typing import Optional
import torch
import torch.nn as nn

from vicentin.deep_learning.utils import mirror_model


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        input_shape: Optional[tuple] = None,
    ):
        super().__init__()
        self.encoder = encoder

        if decoder is None:
            decoder = mirror_model(encoder)

        self.decoder = decoder

        if input_shape is not None:
            param = next(self.encoder.parameters(), None)
            device = param.device if param is not None else "cpu"

            self.eval()
            with torch.no_grad():
                x = torch.zeros(
                    1, *input_shape, dtype=torch.float, device=device
                )
                self.forward(x)
            self.train()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
