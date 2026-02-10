import torch
import torch.nn as nn

from vicentin.deep_learning.models.siamese import SiameseNetwork


class SimCLR(nn.Module):
    def __init__(
        self, encoder: nn.Module, embedding_dim: int, input_shape: tuple
    ):
        super().__init__()

        param = next(encoder.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        x = torch.zeros(1, *input_shape, device=device)

        with torch.no_grad():
            f_dim = encoder(x).flatten(1).shape[1]

        g_dim = embedding_dim

        head = nn.Sequential(
            nn.Linear(f_dim, f_dim),
            nn.BatchNorm1d(f_dim),
            nn.ReLU(),
            nn.Linear(f_dim, g_dim),
        )

        self.siamese_model = SiameseNetwork(encoder, head)

    def forward(self, x):
        return self.siamese_model(x)
