from .BaseLoss import BaseLoss
from .WrapTorchLoss import WrapTorchLoss
from .WeightedSumLoss import WeightedSumLoss

from .GaussianKLDivergenceLoss import GaussianKLDivergenceLoss

from .WassersteinGANLoss import (
    WassersteinDiscriminatorLoss,
    WassersteinGeneratorLoss,
)
from .VAELoss import VAELoss

from .SupInfoNCE import SupInfoNCELoss
