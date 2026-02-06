import torch
import torch.nn as nn
import copy
from typing import List, Optional

from vicentin.deep_learning.utils import init_layer


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module | List[nn.Module] = nn.ReLU(),
        dropout_prob: float | List[float] = 0.0,
        use_batchnorm: bool | List[bool] = False,
        output_activation: Optional[nn.Module] = None,
        init_method: str = "kaiming",
    ):
        super(MLP, self).__init__()

        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "activation": activation,
            "dropout_prob": dropout_prob,
            "use_batchnorm": use_batchnorm,
            "output_activation": output_activation,
            "init_method": init_method,
        }

        num_hidden = len(hidden_dims)

        def to_list(
            param: float | bool | nn.Module | List,
            name: str,
            is_module: bool = False,
        ):
            if isinstance(param, list):
                if len(param) != num_hidden:
                    raise ValueError(
                        f"{name} length ({len(param)}) must match hidden layers ({num_hidden})."
                    )
                return param

            if is_module and isinstance(param, nn.Module):
                return [copy.deepcopy(param) for _ in range(num_hidden)]

            return [param] * num_hidden

        activations_list = to_list(activation, "activation", is_module=True)
        dropouts_list = to_list(dropout_prob, "dropout_prob")
        batchnorms_list = to_list(use_batchnorm, "use_batchnorm")

        layers = []
        current_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            lin = nn.Linear(current_dim, h_dim)
            act = activations_list[i]

            init_layer(
                lin, act, init_method  # pyright: ignore[reportArgumentType]
            )
            layers.append(lin)

            if batchnorms_list[i]:
                layers.append(nn.BatchNorm1d(h_dim))

            layers.append(act)

            if dropouts_list[i] > 0.0:  # pyright: ignore[reportOperatorIssue]
                layers.append(
                    nn.Dropout(
                        dropouts_list[i]  # pyright: ignore[reportArgumentType]
                    )
                )

            current_dim = h_dim

        out_lin = nn.Linear(current_dim, output_dim)
        init_layer(out_lin, output_activation, init_method)
        layers.append(out_lin)

        if output_activation is not None:
            layers.append(output_activation)

        self.model = nn.Sequential(*layers)

    def mirror(self) -> nn.Module:
        c = self.config

        new_hidden = list(reversed(c["hidden_dims"]))

        def reverse_if_list(param):
            if isinstance(param, list):
                return list(reversed(param))
            return param

        return MLP(
            input_dim=c["output_dim"],
            hidden_dims=new_hidden,
            output_dim=c["input_dim"],
            activation=reverse_if_list(c["activation"]),
            dropout_prob=reverse_if_list(c["dropout_prob"]),
            use_batchnorm=reverse_if_list(c["use_batchnorm"]),
            output_activation=None,
            init_method=c["init_method"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
