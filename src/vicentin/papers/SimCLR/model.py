from typing import Tuple, Optional
import torch
import torch.nn as nn

from vicentin.deep_learning.models import SiameseNetwork
from vicentin.deep_learning.blocks import MLP


class SimCLR(nn.Module):
    """
    SimCLR (Simple Framework for Contrastive Learning of Visual Representations) model architecture.
    Supports both v1 and v2 projection head designs using your flexible MLP class.

    Attributes
    ----------
    siamese_model : SiameseNetwork
        The underlying Siamese network that handles the forward pass.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        input_shape: Tuple[int, ...],
        hidden_dim: Optional[int] = None,
        simclr_version: int = 1,
    ) -> None:
        """
        Initialize the SimCLR model.

        Parameters
        ----------
        encoder : nn.Module
            The backbone encoder (e.g., ResNet18).
        embedding_dim : int
            The dimensionality of the final projected latent space (z).
        input_shape : Tuple[int, ...]
            The shape of the input data (C, H, W) used to infer encoder output dim.
        hidden_dim : int, optional
            The dimensionality of the hidden layer in the projection head.
            Defaults to encoder output dimension.
        simclr_version : int, optional
            Version of the SimCLR architecture (1 or 2).
            v1 uses a 2-layer projection head (1 hidden layer).
            v2 uses a 3-layer projection head (2 hidden layers).
            Default is 1.
        """
        super().__init__()

        param = next(encoder.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        dummy_input = torch.zeros(1, *input_shape, device=device)
        with torch.no_grad():
            f_dim = encoder(dummy_input).flatten(1).shape[1]

        h_dim = hidden_dim if hidden_dim is not None else f_dim

        n_layers = 1 if simclr_version == 1 else 2
        mlp_hidden_dims = [h_dim] * n_layers

        head = MLP(f_dim, mlp_hidden_dims, embedding_dim, use_batchnorm=True)

        self.siamese_model = SiameseNetwork(encoder, head)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        return self.siamese_model(x)
