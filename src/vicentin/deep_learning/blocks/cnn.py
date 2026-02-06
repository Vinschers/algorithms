import copy
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicentin.deep_learning.utils import init_layer, mirror_model


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | Literal["auto"] = "auto",
        output_padding: int | Literal["auto"] = "auto",
        dilation: int = 1,
        groups: int = 1,
        bias: bool | Literal["auto"] = "auto",
        transposed: bool = False,
        activation: nn.Module = nn.ReLU(),
        norm_layer: Optional[type[nn.Module]] = nn.BatchNorm2d,
        dropout_prob: float = 0.0,
        init_method: str = "kaiming_fan_out",
        pad_if_odd: bool = True,
    ):
        super().__init__()

        self.config = locals()
        del self.config["self"]
        del self.config["__class__"]

        self.transposed = transposed
        self.act = (
            copy.deepcopy(activation)
            if isinstance(activation, nn.Module)
            else activation
        )
        self.dropout = (
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )
        self.pad_if_odd = pad_if_odd
        self.stride = stride

        final_padding = 0
        final_output_padding = 0

        if padding == "auto":
            if transposed:
                required_diff = kernel_size - stride
                final_output_padding = required_diff % 2
                final_padding = (required_diff + final_output_padding) // 2
            else:
                final_padding = (dilation * (kernel_size - 1)) // 2
        else:
            final_padding = padding
            if output_padding == "auto":
                final_output_padding = 0
            else:
                final_output_padding = output_padding

        if bias == "auto":
            bias = norm_layer is None

        if self.transposed:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=final_padding,
                output_padding=final_output_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=final_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.norm = None
        if norm_layer is not None:
            if isinstance(norm_layer, type):
                if norm_layer == nn.GroupNorm:
                    self.norm = nn.GroupNorm(
                        num_groups=min(32, out_channels),
                        num_channels=out_channels,
                    )
                else:
                    self.norm = norm_layer(out_channels)
            else:
                self.norm = copy.deepcopy(norm_layer)

        init_layer(self.conv, self.act, method=init_method)

    def mirror(self) -> nn.Module:
        """Returns the structural inverse of this block."""
        c = self.config

        return ConvBlock(
            in_channels=c["out_channels"],
            out_channels=c["in_channels"],
            kernel_size=c["kernel_size"],
            stride=c["stride"],
            padding=c["padding"],
            output_padding=c["output_padding"],
            dilation=c["dilation"],
            groups=c["groups"],
            bias=c["bias"],
            transposed=not c["transposed"],
            activation=copy.deepcopy(c["activation"]),
            norm_layer=c["norm_layer"],
            dropout_prob=c["dropout_prob"],
            init_method=c["init_method"],
            pad_if_odd=c["pad_if_odd"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.transposed and self.pad_if_odd and self.stride > 1:
            h, w = x.shape[-2:]
            pad_h = h % 2 != 0
            pad_w = w % 2 != 0
            if pad_h or pad_w:
                x = F.pad(x, (0, int(pad_w), 0, int(pad_h)))

        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_sizes: int | List[int] = 3,
        strides: int | List[int] = 1,
        transposed: bool | List[bool] = False,
        paddings: int | List[int] | Literal["auto"] = "auto",
        activations: nn.Module | List[nn.Module] = nn.ReLU(),
        dropout_probs: float | List[float] = 0.0,
        use_batchnorm: bool | List[bool] = True,
    ):
        super().__init__()

        num_layers = len(hidden_channels)

        def to_list(param, name, is_module=False):
            if isinstance(param, list):
                if len(param) != num_layers:
                    raise ValueError(
                        f"{name} length must match hidden_channels ({num_layers})"
                    )
                return param

            if is_module and isinstance(param, nn.Module):
                return [copy.deepcopy(param) for _ in range(num_layers)]

            return [param] * num_layers

        k_list = to_list(kernel_sizes, "kernel_sizes")
        s_list = to_list(strides, "strides")
        t_list = to_list(transposed, "transposed")
        p_list = to_list(paddings, "paddings")
        drop_list = to_list(dropout_probs, "dropout_probs")
        bn_list = to_list(use_batchnorm, "use_batchnorm")
        act_list = to_list(activations, "activations", is_module=True)

        layers = []
        current_in = in_channels

        for i, h_dim in enumerate(hidden_channels):
            norm_type = nn.BatchNorm2d if bn_list[i] else None

            layers.append(
                ConvBlock(
                    in_channels=current_in,
                    out_channels=h_dim,
                    kernel_size=k_list[
                        i
                    ],  # pyright: ignore[reportArgumentType]
                    stride=s_list[i],  # pyright: ignore[reportArgumentType]
                    padding=p_list[i],  # pyright: ignore[reportArgumentType]
                    transposed=t_list[i],  # pyright: ignore[reportArgumentType]
                    activation=act_list[i],
                    norm_layer=norm_type,
                    dropout_prob=drop_list[
                        i
                    ],  # pyright: ignore[reportArgumentType]
                )
            )
            current_in = h_dim

        self.net = nn.Sequential(*layers)

    def mirror(self):
        return mirror_model(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
