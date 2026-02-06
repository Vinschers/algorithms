from typing import Optional, Tuple
import torch
import torch.nn as nn

from vicentin.deep_learning.utils import mirror_model, init_layer


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        decoder: Optional[nn.Module] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim

        if decoder is None:
            self.decoder = mirror_model(encoder)
        else:
            self.decoder = decoder

        self.fc_mu: Optional[nn.Linear] = None
        self.fc_var: Optional[nn.Linear] = None
        self.fc_decode: Optional[nn.Linear] = None
        self.last_feature_shape: Optional[torch.Size] = None

        if input_shape is not None:
            self._configure_architecture(input_shape)

    def _configure_architecture(self, input_shape: Tuple[int, ...]):
        param = next(self.encoder.parameters(), None)
        device = param.device if param is not None else "cpu"

        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, *input_shape, device=device)

            features = self.encoder(x)

            self.last_feature_shape = features.shape[1:]
            if self.last_feature_shape is None:
                raise RuntimeError("Feature shape does not exist.")

            flat_dim = features.flatten(1).shape[1]

            self.fc_mu = nn.Linear(flat_dim, self.latent_dim).to(device)
            self.fc_var = nn.Linear(flat_dim, self.latent_dim).to(device)
            self.fc_decode = nn.Linear(self.latent_dim, flat_dim).to(device)

            init_layer(self.fc_mu, None)
            init_layer(self.fc_var, None)
            init_layer(self.fc_decode, None)

            z = torch.randn(1, self.latent_dim, device=device)
            feat_recon = self.fc_decode(z).view(1, *self.last_feature_shape)
            self.decoder(feat_recon)

        self.train()

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor):
        if (
            self.fc_mu is None
            or self.fc_var is None
            or self.fc_decode is None
            or self.last_feature_shape is None
        ):
            raise RuntimeError(
                "VAE architecture not configured! Pass 'input_shape' to __init__."
            )

        features = self.encoder(x)

        flat_features = features.flatten(1)

        mu = self.fc_mu(flat_features)
        logvar = self.fc_var(flat_features)

        z = self.reparameterize(mu, logvar)

        recon_features_flat = self.fc_decode(z)

        recon_features = recon_features_flat.view(
            x.shape[0], *self.last_feature_shape
        )

        x_recon = self.decoder(recon_features)

        return x_recon, mu, logvar
