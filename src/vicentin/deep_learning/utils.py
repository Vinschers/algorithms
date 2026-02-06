import copy
from typing import Optional

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def init_layer(
    layer: nn.Module,
    activation: Optional[nn.Module] = None,
    method: str = "auto",
):
    if not hasattr(layer, "weight") or layer.weight is None:
        return

    if not isinstance(layer.weight, torch.Tensor):
        return

    nonlinearity = "linear"
    param = 0.0

    if isinstance(activation, nn.Module):
        if isinstance(activation, (nn.ReLU, nn.ReLU6)):
            nonlinearity = "relu"
        elif isinstance(activation, nn.LeakyReLU):
            nonlinearity = "leaky_relu"
            param = activation.negative_slope
        elif isinstance(activation, (nn.Tanh, nn.Tanhshrink)):
            nonlinearity = "tanh"
        elif isinstance(activation, (nn.Sigmoid, nn.LogSigmoid)):
            nonlinearity = "sigmoid"
        elif isinstance(activation, (nn.GELU, nn.SiLU)):
            # GELU/SiLU are close enough to ReLU for Kaiming init purposes
            nonlinearity = "relu"
        elif isinstance(activation, nn.PReLU):
            nonlinearity = "leaky_relu"
            # PReLU weight is a learnable tensor; we use the mean for initialization estimate
            param = activation.weight.data.mean().item()
        elif isinstance(activation, nn.SELU):
            nonlinearity = "selu"

    if method == "auto":
        if nonlinearity in ["relu", "leaky_relu"]:
            if isinstance(
                layer,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                method = "kaiming_fan_out"
            else:
                method = "kaiming_fan_in"
        else:
            method = "xavier"

    if method == "kaiming_fan_out":
        nn.init.kaiming_normal_(
            layer.weight, mode="fan_out", nonlinearity=nonlinearity, a=param
        )

    elif method == "kaiming_fan_in":
        nn.init.kaiming_normal_(
            layer.weight, mode="fan_in", nonlinearity=nonlinearity, a=param
        )

    elif method == "xavier":
        try:
            gain = nn.init.calculate_gain(nonlinearity, param)
        except ValueError:
            gain = 1.0
        nn.init.xavier_uniform_(layer.weight, gain=gain)

    elif method == "orthogonal":
        try:
            gain = nn.init.calculate_gain(nonlinearity, param)
        except ValueError:
            gain = 1.0
        nn.init.orthogonal_(layer.weight, gain=gain)

    if (
        hasattr(layer, "bias")
        and layer.bias is not None
        and isinstance(layer.bias, torch.Tensor)
    ):
        nn.init.constant_(layer.bias, 0)


def mirror_layer(layer: nn.Module) -> nn.Module:
    if isinstance(layer, nn.Linear):
        return nn.Linear(
            in_features=layer.out_features,
            out_features=layer.in_features,
            bias=layer.bias is not None,
        )

    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        dim = (
            1
            if isinstance(layer, nn.Conv1d)
            else (2 if isinstance(layer, nn.Conv2d) else 3)
        )
        TransposedCls = getattr(nn, f"ConvTranspose{dim}d")

        s = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride,) * dim
        )
        k = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size,) * dim
        )
        op = tuple((k_i - s_i) % 2 if s_i > 1 else 0 for k_i, s_i in zip(k, s))
        if len(op) == 1:
            op = op[0]

        return TransposedCls(
            in_channels=layer.out_channels,
            out_channels=layer.in_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=op,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
        )

    if isinstance(
        layer, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    ):
        dim = (
            1
            if isinstance(layer, nn.ConvTranspose1d)
            else (2 if isinstance(layer, nn.ConvTranspose2d) else 3)
        )
        ConvCls = getattr(nn, f"Conv{dim}d")

        return ConvCls(
            in_channels=layer.out_channels,
            out_channels=layer.in_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
        )

    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return type(layer)(
            layer.num_features,
            eps=layer.eps,
            momentum=layer.momentum,
            affine=layer.affine,
        )

    if isinstance(
        layer, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    ):
        return type(layer)(
            layer.num_features,
            eps=layer.eps,
            momentum=layer.momentum,  # pyright: ignore[reportArgumentType]
            affine=layer.affine,
        )

    if isinstance(layer, nn.GroupNorm):
        return nn.GroupNorm(
            layer.num_groups,
            layer.num_channels,
            eps=layer.eps,
            affine=layer.affine,
        )

    if isinstance(layer, nn.LayerNorm):
        return nn.LayerNorm(
            layer.normalized_shape,  # pyright: ignore[reportArgumentType]
            eps=layer.eps,
            elementwise_affine=layer.elementwise_affine,
        )

    if isinstance(
        layer,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
        ),
    ):
        stride = layer.stride if layer.stride is not None else layer.kernel_size
        return nn.Upsample(scale_factor=stride, mode="nearest")

    if isinstance(layer, nn.Upsample):
        sf = layer.scale_factor if layer.scale_factor else 2
        dim = 2
        PoolCls = nn.AvgPool2d
        return PoolCls(
            kernel_size=int(sf),  # pyright: ignore[reportArgumentType]
            stride=int(sf),  # pyright: ignore[reportArgumentType]
        )

    if isinstance(
        layer,
        (
            nn.ReLU,
            nn.LeakyReLU,
            nn.PReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.ELU,
            nn.SELU,
            nn.CELU,
            nn.Sigmoid,
            nn.Tanh,
            nn.Softmax,
            nn.LogSoftmax,
            nn.Dropout,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
        ),
    ):
        return copy.deepcopy(layer)

    if isinstance(layer, nn.Identity):
        return nn.Identity()

    print(
        f"Warning: layer {type(layer).__name__} not explicitly mirrored. Returning copy."
    )
    return copy.deepcopy(layer)


def mirror_model(model: nn.Module) -> nn.Module:
    """
    Recursively creates a mirror (inverse) of any PyTorch model.
    1. If model has .mirror(), use it.
    2. If model is Sequential, reverse and mirror children.
    3. If model is a primitive (Linear/Conv), manually invert.
    4. Otherwise, deepcopy (Activations, Norms).
    """
    if hasattr(model, "mirror") and callable(model.mirror):
        return model.mirror()

    if isinstance(model, (nn.Sequential, nn.ModuleList)):
        children = list(model.children())
        mirrored = [mirror_model(c) for c in reversed(children)]

        if isinstance(model, nn.Sequential):
            return nn.Sequential(*mirrored)
        return nn.ModuleList(mirrored)

    return mirror_layer(model)
