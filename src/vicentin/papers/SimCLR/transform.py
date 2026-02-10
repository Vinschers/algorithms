from typing import Tuple, Optional
import torch
from torchvision import transforms


class SimCLRTransform:
    """
    A stochastic data augmentation module for SimCLR.

    This transform generates two augmented views of the same image. The pipeline includes
    random resized cropping, horizontal flipping, color jittering, random grayscale,
    Gaussian blurring, and normalization.

    Attributes
    ----------
    transform : transforms.Compose
        The composed torchvision transform pipeline.
    """

    def __init__(
        self,
        img_size: int,
        s: float = 1.0,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Initialize the SimCLR transformation pipeline.

        Parameters
        ----------
        img_size : int
            The size (height/width) of the output image.
        s : float, optional
            Strength of the color jitter augmentation. Default is 1.0.
        mean : Tuple[float, float, float], optional
            Mean for normalization. Defaults to ImageNet values (0.485, 0.456, 0.406).
        std : Tuple[float, float, float], optional
            Standard deviation for normalization. Defaults to ImageNet values (0.229, 0.224, 0.225).
        """
        self.mean = mean or (0.485, 0.456, 0.406)
        self.std = std or (0.229, 0.224, 0.225)

        kernel_size = int(0.1 * img_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the transformation to an input image twice.

        Parameters
        ----------
        x : Any
            Input image (PIL Image or Tensor).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two different augmented views of the input image.
        """
        return self.transform(x), self.transform(x)
