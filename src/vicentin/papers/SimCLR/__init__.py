from .transform import SimCLRTransform
from .model import SimCLR
from .trainer import SimCLRTrainer

from typing import Tuple
import torch.nn as nn
from torchvision.models import resnet18, resnet50


def create_simclr_resnet(
    arch: str = "resnet18",
    embedding_dim: int = 128,
    simclr_version: int = 2,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    pretrained_backbone: bool = False,
) -> Tuple[SimCLR, SimCLRTransform]:
    """
    Factory function to create a SimCLR model with a ResNet backbone and
    its corresponding data transform.

    Args:
        arch: 'resnet18' or 'resnet50'.
        embedding_dim: Dimension of the projection head output.
        simclr_version: 1 or 2 (depth of projection head).
        input_shape: (C, H, W) of the input images.
        pretrained_backbone: If True, uses ImageNet weights for the ResNet encoder.

    Returns:
        model: The configured SimCLR module.
        transform: The SimCLR augmentation pipeline ready for a DataLoader.
    """

    weights = "DEFAULT" if pretrained_backbone else None
    if arch == "resnet18":
        backbone = resnet18(weights=weights)
    elif arch == "resnet50":
        backbone = resnet50(weights=weights)
    else:
        raise ValueError(f"Architecture {arch} not supported.")

    backbone.fc = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]

    model = SimCLR(backbone, embedding_dim, input_shape, simclr_version)

    img_size = input_shape[1]
    transform = SimCLRTransform(img_size=img_size)

    return model, transform
