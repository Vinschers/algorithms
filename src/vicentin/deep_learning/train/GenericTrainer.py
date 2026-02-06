import math
import os
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from torch.amp import GradScaler  # pyright: ignore[reportPrivateImportUsage]
from tqdm.auto import tqdm

from vicentin.deep_learning.utils import get_device


class GenericTrainer(ABC):
    def __init__(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        schedulers: Optional[
            Dict[str, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        float16: bool = False,
    ):
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers if schedulers else {}
        self.hyperparams = hyperparams if hyperparams else {}

        self.device = get_device() if device is None else device

        self.amp = float16
        self.scaler = GradScaler(enabled=self.amp)

        for name, model in self.models.items():
            self.models[name] = model.to(self.device)

        self.history = defaultdict(list)
        self.current_epoch = 0

    @abstractmethod
    def train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Implementation specific logic for one training step.
        Must return a dict of metrics (e.g., {'loss': 0.5}).
        """
        pass

    @abstractmethod
    def validate_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Implementation specific logic for one validation step.
        Must return a dict of metrics.
        """
        pass

    def optimize(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        gradient_clipping: Optional[float] = None,
    ):
        """
        Unified helper to handle Backward + Step.
        Automatically handles AMP scaling if self.use_amp is True.
        Call this inside your train_step instead of loss.backward().
        """

        optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(optimizer)
        if gradient_clipping is not None:
            params = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(params, max_norm=gradient_clipping)

        self.scaler.step(optimizer)
        self.scaler.update()

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        return batch

    def _set_model_mode(self, training: bool):
        for model in self.models.values():
            model.train() if training else model.eval()

    def _run_epoch_loop(
        self, loader, step_method, desc: str, training: bool
    ) -> Dict[str, float]:
        self._set_model_mode(training)
        metrics_accumulator = defaultdict(float)
        steps = 0

        grad_context = torch.enable_grad() if training else torch.no_grad()

        with grad_context:
            pbar = tqdm(loader, desc=desc, leave=True)
            for batch_idx, batch in enumerate(pbar):
                batch = self._move_to_device(batch)

                with autocast(enabled=self.amp, device_type=self.device.type):
                    metrics = step_method(batch, batch_idx)

                for k, v in metrics.items():
                    metrics_accumulator[k] += v
                steps += 1

                pbar.set_postfix(metrics)

        return {k: v / steps for k, v in metrics_accumulator.items()}

    def _update_history(self, train_metrics, val_metrics):
        val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
        all_metrics = {**train_metrics, **val_metrics_prefixed}

        for k, v in all_metrics.items():
            self.history[k].append(v)

        return all_metrics

    def _handle_checkpoint(
        self,
        epoch: int,
        all_metrics: Dict[str, float],
        path: str,
        mode: Union[str, int],
        monitor: str,
        current_best: float,
    ) -> float:
        new_best = current_best

        if isinstance(mode, int) and (epoch + 1) % mode == 0:
            root, ext = os.path.splitext(path)
            save_path = f"{root}_epoch_{epoch+1}{ext}"
            self.save_checkpoint(save_path, epoch, current_best)
            print(f"  >>> Saved Checkpoint: {save_path}")
            return current_best

        elif isinstance(mode, str) and monitor in all_metrics:
            score = all_metrics[monitor]

            improved = False
            if mode == "min" and score < current_best:
                improved = True
            elif mode == "max" and score > current_best:
                improved = True

            if improved:
                print(
                    f"  >>> New Best {monitor}: {score:.4f} (was {current_best:.4f})"
                )
                self.save_checkpoint(path, epoch, score)
                new_best = score

        return new_best

    def save_checkpoint(self, path: str, epoch: int, best_score: float):
        torch.save(
            {
                "epoch": epoch,
                "hyperparams": self.hyperparams,
                "history": dict(self.history),
                "best_score": best_score,
                "models": {k: v.state_dict() for k, v in self.models.items()},
                "optimizers": {
                    k: v.state_dict() for k, v in self.optimizers.items()
                },
                "scaler": self.scaler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        for name, model in self.models.items():
            if name in checkpoint["models"]:
                model.load_state_dict(checkpoint["models"][name])
        for name, opt in self.optimizers.items():
            if name in checkpoint["optimizers"]:
                opt.load_state_dict(checkpoint["optimizers"][name])

        self.history = defaultdict(list, checkpoint.get("history", {}))
        self.hyperparams = checkpoint.get("hyperparams", {})

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        return checkpoint.get("epoch", 0) + 1

    def fit(
        self,
        train_loader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        monitor: Optional[str] = "val_loss",
        mode: Union[str, int] = "min",
        resume_from: Optional[str] = None,
    ):
        """
        The Orchestrator.

        Args:
            mode: "min" (save on min metric), "max" (save on max metric), or int (save every N epochs).
            resume_from: Path to a checkpoint to resume training.
        """

        start_epoch = self.load_checkpoint(resume_from) if resume_from else 0

        if mode == "min":
            best_score = math.inf
        elif mode == "max":
            best_score = -math.inf
        else:
            best_score = 0

        if monitor in self.history and self.history[monitor]:
            if mode == "min":
                best_score = min(self.history[monitor])
            elif mode == "max":
                best_score = max(self.history[monitor])

        if start_epoch >= epochs:
            print("Training already completed.")
            return dict(self.history)

        print(
            f"Training on {self.device} [AMP={self.amp}] from epoch {start_epoch+1} to {epochs}..."
        )

        for epoch in range(start_epoch, epochs):

            self.current_epoch = epoch

            train_metrics = self._run_epoch_loop(
                loader=train_loader,
                step_method=self.train_step,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                training=True,
            )

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._run_epoch_loop(
                    loader=val_loader,
                    step_method=self.validate_step,
                    desc=f"Epoch {epoch+1}/{epochs} [Val]",
                    training=False,
                )

            all_metrics = self._update_history(train_metrics, val_metrics)

            for scheduler in self.schedulers.values():
                if isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    metric_to_monitor = all_metrics.get(monitor, 0.0)
                    scheduler.step(metric_to_monitor)
                else:
                    scheduler.step()

            log_str = " | ".join(
                [f"{k}: {v:.4f}" for k, v in all_metrics.items()]
            )
            print(f"End of Epoch {epoch+1}: {log_str}")

            if checkpoint_path is not None and monitor is not None:
                best_score = self._handle_checkpoint(
                    epoch,
                    all_metrics,
                    checkpoint_path,
                    mode,
                    monitor,
                    best_score,
                )

        return dict(self.history)
