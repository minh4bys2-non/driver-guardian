import logging
import math
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import TrainConfig

_log = logging.getLogger("train_")

class TrainingLogger:
    """TensorBoard logger for training, model and system metrics."""

    def __init__(
        self,
        writer: SummaryWriter,
        cfg: TrainConfig,
    ):
        self.writer = writer
        self.log_interval = cfg.log_interval
        self.histogram_interval = cfg.log_hist_interval
        self.log_gradients_enabled = cfg.log_gradients
        self.log_weights_enabled = cfg.log_weights
        self._grad_norm_buffer = deque(maxlen=cfg.log_window_size)

    def should_log_scalar(self, step: int) -> bool:
        return self.log_interval > 0 and step % self.log_interval == 0

    def should_log_hist(self, step: int) -> bool:
        return self.histogram_interval > 0 and step % self.histogram_interval == 0

    @staticmethod
    def snapshot_params(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def log_losses(self, items: dict, step: int, phase: str = "train_") -> None:
        if phase not in {"train_", "val", "finetune_train", "finetune_val"}:
            raise ValueError(f"Invalid phase: {phase}")
        groups = {
            "loss_total": {
                "total": items["loss"],
                "o2m": items["loss_o2m"],
                "o2o": items["loss_o2o"],
            },
            "loss_o2m_parts": {
                key: items[f"o2m/{key}"] for key in ("iou", "cls", "dfl")
            },
            "loss_o2o_parts": {
                key: items[f"o2o/{key}"] for key in ("iou", "cls", "dfl")
            },
            "n_pos": {key: items[f"{key}/n_pos"] for key in ("o2m", "o2o")},
        }
        for name, values in groups.items():
            self.writer.add_scalars(f"{phase}/{name}", values, step)

    def log_loss_ratios(self, items: dict, step: int, phase: str = "train_") -> None:
        total = items["loss"] + 1e-8
        self.writer.add_scalars(
            f"{phase}/loss_ratios",
            {
                "o2m_ratio": items["loss_o2m"] / total,
                "o2o_ratio": items["loss_o2o"] / total,
            },
            step,
        )
        o2m_total = items["loss_o2m"] + 1e-8
        self.writer.add_scalars(
            f"{phase}/o2m_component_ratios",
            {
                key: items[f"o2m/{key}"] / o2m_total
                for key in ("iou", "cls", "dfl")
            },
            step,
        )

    def log_scalars(self, scalars: dict, step: int) -> None:
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)

    def log_gradients(
        self,
        model: nn.Module,
        step: int,
        total_norm: Optional[float] = None,
    ) -> float:
        if not self.log_gradients_enabled:
            return total_norm if total_norm is not None else 0.0
        do_scalar = self.should_log_scalar(step)
        do_hist = self.should_log_hist(step)
        if not (do_scalar or do_hist):
            return total_norm if total_norm is not None else 0.0

        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None or not grad.numel():
                continue
            if do_hist:
                self._log_histogram(f"Gradients/{name}", grad, step, "gradient")
            if do_scalar:
                rms = grad.norm().item() / math.sqrt(grad.numel())
                self.writer.add_scalar(f"Gradients_RMS/{name}", rms, step)

        if do_scalar and total_norm is not None:
            norm = float(total_norm)
            self._grad_norm_buffer.append(norm)
            self.writer.add_scalar("Gradients/total_norm", norm, step)
            self.writer.add_scalar(
                "Gradients/avg_norm",
                sum(self._grad_norm_buffer) / len(self._grad_norm_buffer),
                step,
            )
        return total_norm if total_norm is not None else 0.0

    def log_weights(self, model: nn.Module, step: int) -> None:
        if not self.log_weights_enabled:
            return
        do_scalar = self.should_log_scalar(step)
        do_hist = self.should_log_hist(step)
        if not (do_scalar or do_hist):
            return

        for name, param in model.named_parameters():
            weight = param.detach()
            if do_hist:
                self._log_histogram(f"Weights/{name}", weight, step, "weight")
            if do_scalar and weight.numel():
                values = {
                    "mean": weight.mean(),
                    "std": weight.std(unbiased=False),
                    "rms": weight.norm() / math.sqrt(weight.numel()),
                    "max": weight.max(),
                    "min": weight.min(),
                }
                for metric, value in values.items():
                    self.writer.add_scalar(
                        f"Weights_Stats/{name}/{metric}", value.item(), step
                    )

    def _log_histogram(
        self, tag: str, tensor: torch.Tensor, step: int, kind: str
    ) -> None:
        if torch.isfinite(tensor).all():
            self.writer.add_histogram(tag, tensor, step)
        else:
            _log.warning("Skipping non-finite %s histogram '%s' at step %s", kind, tag, step)

    def log_weight_updates(
        self,
        model: nn.Module,
        prev_params: Dict[str, torch.Tensor],
        step: int,
    ) -> None:
        if not self.should_log_scalar(step) or prev_params is None:
            return
        for name, param in model.named_parameters():
            previous = prev_params.get(name)
            if previous is None:
                continue
            update = param.detach() - previous
            ratio = (update.abs() / (previous.abs() + 1e-8)).mean()
            self.writer.add_scalar(f"Update_Ratio/{name}", ratio.item(), step)
            self.writer.add_scalar(
                f"Update_Magnitude/{name}", update.norm().item(), step
            )

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        if not self.should_log_scalar(step):
            return
        for index, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"Learning_Rate/group_{index}", group["lr"], step)
            if "weight_decay" in group:
                self.writer.add_scalar(
                    f"Weight_Decay/group_{index}", group["weight_decay"], step
                )
        if epoch is not None:
            self.writer.add_scalar("Training/epoch", epoch, step)

    def log_ema(self, ema, step: int) -> None:
        if ema is None or not self.should_log_scalar(step):
            return
        self.writer.add_scalar("EMA/current_decay", ema._current_decay(), step)
        self.writer.add_scalar("EMA/updates", ema.updates, step)
        self.writer.add_scalar(
            "EMA/warmup_progress",
            min(ema.updates / max(1, ema.warmup_updates), 1.0),
            step,
        )

    def log_ema_params(
        self,
        ema_model: Optional[nn.Module],
        step: int,
        prefix: str = "EMA",
    ) -> None:
        if ema_model is None or not self.should_log_scalar(step):
            return
        norms = [
            param.detach().norm().item() ** 2
            for param in ema_model.parameters()
            if param.dtype.is_floating_point
        ]
        self.writer.add_scalar(f"{prefix}/param_norm", sum(norms) ** 0.5, step)
        self.writer.add_scalar(f"{prefix}/param_count", len(norms), step)

    def log_gpu_memory(self, step: int) -> None:
        if not torch.cuda.is_available() or not self.should_log_scalar(step):
            return
        gb = 1024**3
        allocated = torch.cuda.memory_allocated() / gb
        reserved = torch.cuda.memory_reserved() / gb
        values = {
            "GPU_memory_allocated_GB": allocated,
            "GPU_memory_reserved_GB": reserved,
            "GPU_max_memory_allocated_GB": torch.cuda.max_memory_allocated() / gb,
            "GPU_memory_utilization": allocated / (reserved + 1e-8),
        }
        for name, value in values.items():
            self.writer.add_scalar(f"System/{name}", value, step)

    def log_batchnorm(self, model: nn.Module, step: int) -> None:
        if not self.should_log_hist(step):
            return
        for name, module in model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            tag = f"BN/{name.replace('.', '/')}"
            values = {}
            if module.track_running_stats and module.running_mean is not None:
                values["running_mean"] = module.running_mean.mean()
                values["running_var"] = module.running_var.mean()
            if module.weight is not None:
                values |= {
                    "gamma_mean": module.weight.mean(),
                    "gamma_std": module.weight.std(unbiased=False),
                }
            if module.bias is not None:
                values |= {
                    "beta_mean": module.bias.mean(),
                    "beta_std": module.bias.std(unbiased=False),
                }
            for metric, value in values.items():
                self.writer.add_scalar(f"{tag}/{metric}", value.item(), step)

    def log_hparams(self, cfg, step: int = 0) -> None:
        groups = {
            "Model": ("nc", "reg_max", "backbone_w", "backbone_n", "neck_n", "strides"),
            "Training": (
                "epochs", "batch_size", "img_size", "val_interval_steps",
                "save_ckpt_interval_steps", "lr0", "lr_min_factor", "warmup_epochs",
                "weight_decay", "grad_clip_norm", "optimizer",
            ),
            "Loss": (
                "cls_gain", "box_gain", "dfl_gain", "w_o2o", "w_o2m",
                "topk_o2m", "topk_o2o", "alpha", "beta",
            ),
            "EMA": ("use_ema", "ema_decay", "ema_warmup_updates"),
        }
        for group, names in groups.items():
            text = "\n".join(
                f"- {name}: {getattr(cfg, name, 'N/A')}" for name in names
            )
            self.writer.add_text(f"Hyperparameters/{group}", text, step)


class TimeTracker:
    def __init__(self, writer: SummaryWriter, cfg: TrainConfig):
        self.writer = writer
        self.batch_times = deque(maxlen=cfg.log_window_size)

    def log_batch_time(self, batch_time: float, step: int) -> None:
        self.batch_times.append(batch_time)
        self.writer.add_scalar("Time/batch_time_ms", batch_time * 1000, step)
        self.writer.add_scalar(
            "Time/avg_batch_time_ms",
            sum(self.batch_times) / len(self.batch_times) * 1000,
            step,
        )

    def log_throughput(self, batch_size: int, batch_time: float, step: int) -> None:
        self.writer.add_scalar(
            "Time/throughput_samples_per_sec", batch_size / batch_time, step
        )


class ActivationTracker:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.hooks = []
        self.step = 0

    def register_hooks(self, model: nn.Module) -> None:
        def make_hook(name):
            def hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self._log_activation(name, output, self.step)
            return hook

        tracked = (nn.Conv2d, nn.BatchNorm2d, nn.SiLU)
        for name, module in model.named_modules():
            if isinstance(module, tracked):
                self.hooks.append(module.register_forward_hook(make_hook(name)))

    def _log_activation(self, name: str, tensor: torch.Tensor, step: int) -> None:
        if not tensor.numel():
            return
        tag = f"Activations/{name.replace('.', '/')}"
        values = {
            "mean": tensor.mean(),
            "std": tensor.std(unbiased=False),
            "max": tensor.max(),
            "min": tensor.min(),
        }
        for metric, value in values.items():
            self.writer.add_scalar(f"{tag}/{metric}", value.item(), step)

    def set_step(self, step: int) -> None:
        self.step = step

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class LossSmoother:
    def __init__(self, cfg: TrainConfig):
        self.window = cfg.log_window_size
        self.buffer = deque(maxlen=self.window)

    def update(self, loss: float) -> float:
        self.buffer.append(loss)
        return sum(self.buffer) / len(self.buffer)

    def log_smoothed(
        self,
        writer: SummaryWriter,
        loss: float,
        step: int,
        tag: str = "loss/smoothed",
    ) -> float:
        smoothed = self.update(loss)
        writer.add_scalar(tag, smoothed, step)
        return smoothed


def log_lr_schedule(writer: SummaryWriter, scheduler, step: int) -> None:
    writer.add_scalar("LR_Schedule/current", scheduler.get_last_lr()[0], step)
    warmup_steps = getattr(scheduler, "warmup_steps", 0)
    if warmup_steps:
        writer.add_scalar(
            "LR_Schedule/warmup_progress", min(step / warmup_steps, 1.0), step
        )


def log_activation_histograms(
    writer: SummaryWriter,
    activations: Dict[str, torch.Tensor],
    step: int,
) -> None:
    for name, activation in activations.items():
        if isinstance(activation, torch.Tensor):
            writer.add_histogram(
                f"Activations_hist/{name.replace('.', '/')}", activation, step
            )
