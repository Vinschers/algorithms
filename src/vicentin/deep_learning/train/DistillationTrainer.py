import torch

from vicentin.deep_learning.train import SupervisedTrainer


class DistillationTrainer(SupervisedTrainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()

    def compute_loss(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch, None

        student_pred = self.models["model"](x)

        with torch.no_grad():
            teacher_pred = self.teacher(x)

        loss, metrics = self.loss_fn(student_pred, teacher_pred, y)

        if "loss" not in metrics:
            metrics["loss"] = loss.item()

        return loss, metrics
