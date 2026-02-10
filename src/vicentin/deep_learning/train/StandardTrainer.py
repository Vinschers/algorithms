import torch

from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

from vicentin.deep_learning.train import GenericTrainer


class StandardTrainer(GenericTrainer, ABC):

    @abstractmethod
    def compute_loss(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    def compute_val_loss(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        return self.compute_loss(batch, batch_idx)

    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        loss, metrics = self.compute_loss(batch, batch_idx)

        for opt in self.optimizers.values():
            self.optimize(loss, opt)

        results = {**metrics}
        if "loss" not in results:
            results["loss"] = loss.item()
        return results

    def validate_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        loss, metrics = self.compute_val_loss(batch, batch_idx)

        results = {**metrics}
        if "loss" not in results:
            results["loss"] = loss.item()
        return results
