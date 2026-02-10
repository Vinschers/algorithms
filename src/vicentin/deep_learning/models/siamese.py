from typing import Optional, Tuple
import torch
import torch.nn as nn

from vicentin.deep_learning.utils import init_layer


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        representation: nn.Module,
        head: Optional[nn.Module] = None,
        embedding_dim: Optional[int] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        representation_partner: Optional[nn.Module] = None,
        head_partner: Optional[nn.Module] = None,
    ):
        """
        Generic Siamese Network supporting both Symmetric and Asymmetric (Teacher-Student) modes.

        Args:
            representation: The primary encoder (student/shared).
            representation_partner: Optional secondary encoder (teacher).
                                    If None, the network is treated as Symmetric (shared weights).
            head: Optional projection head for the primary branch.
            head_partner: Optional projection head for the secondary branch.
                          If None and representation_partner exists, it mirrors 'head' structure.
            embedding_dim: Required if 'head' is None to build the default linear layer.
            input_shape: Input shape (C, H, W) for dynamic configuration.
        """
        super().__init__()
        self.representation = representation
        self.representation_partner = representation_partner
        self.head = head
        self.head_partner = head_partner
        self.embedding_dim = embedding_dim

        self.is_symmetric = representation_partner is None

        if input_shape is not None:
            self._configure_architecture(input_shape)

    def _configure_branch(
        self,
        rep_module: nn.Module,
        head_module: Optional[nn.Module],
        input_shape: Tuple[int, ...],
        device: torch.device,
    ) -> nn.Module:

        x = torch.zeros(1, *input_shape, device=device)
        features = rep_module(x)
        flat_dim = features.flatten(1).shape[1]

        if head_module is None:
            if self.embedding_dim is None:
                new_head = nn.Identity()
            else:
                new_head = nn.Linear(flat_dim, self.embedding_dim).to(device)

            init_layer(new_head, None)
            return new_head
        else:
            flat_features = features.flatten(1)
            head_module(flat_features)

            return head_module

    def _configure_architecture(self, input_shape: Tuple[int, ...]):
        param = next(self.representation.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        self.eval()
        with torch.no_grad():
            self.head = self._configure_branch(
                self.representation, self.head, input_shape, device
            )

            if (
                not self.is_symmetric
                and self.representation_partner is not None
            ):
                self.head_partner = self._configure_branch(
                    self.representation_partner,
                    self.head_partner,
                    input_shape,
                    device,
                )

        self.train()

    def _forward_branch(
        self, rep_module: nn.Module, head_module: nn.Module, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        features = rep_module(x)

        flat_features = features.flatten(1)
        projection = head_module(flat_features)

        return features, projection

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ]
    ):
        if self.head is None:
            raise RuntimeError(
                "Architecture not configured! Pass 'input_shape' to __init__."
            )

        if self.is_symmetric:
            if x2 is not None:
                raise ValueError(
                    "This network is configured as Symmetric (one branch). "
                    "Forward accepts only 'x1'. To process pairs, call forward twice."
                )
            return self._forward_branch(self.representation, self.head, x1)

        else:
            if x2 is None:
                raise ValueError(
                    "This network is configured as Asymmetric (two branches). "
                    "Forward requires both 'x1' and 'x2'."
                )

            assert self.representation_partner is not None
            assert self.head_partner is not None

            out1 = self._forward_branch(self.representation, self.head, x1)
            out2 = self._forward_branch(
                self.representation_partner, self.head_partner, x2
            )

            return out1, out2
