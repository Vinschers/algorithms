from typing import Optional
import torch
import torch.nn.functional as F

from vicentin.deep_learning.loss import BaseLoss


class SupInfoNCELoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 1,
        epsilon: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()

        self.temperature = temperature
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(
        self, *views: torch.Tensor, targets: Optional[torch.Tensor] = None
    ):
        device = views[0].device
        batch_size = views[0].shape[0]

        features = torch.cat(views, dim=0)
        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T)
        logits.div_(self.temperature)

        if targets is not None:
            labels = targets.repeat(len(views))
        else:
            labels = torch.cat(
                [torch.arange(batch_size) for _ in range(len(views))], dim=0
            ).to(device)

        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(labels_mask.shape[0], device=device).bool()

        mask_positives = (labels_mask.bool()) & (~mask_self)

        if self.epsilon > 0:
            logits.sub_(self.epsilon * mask_positives)

        logits.fill_diagonal_(float("-inf"))

        denominator = torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = logits - denominator

        loss = -log_prob[mask_positives]

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        metrics = {"tau": self.temperature, "epsilon": self.epsilon}

        return loss, metrics
