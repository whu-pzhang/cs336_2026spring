"""Stateful training loop."""

from __future__ import annotations

import logging
import time
from typing import cast

import torch

from cs336_basics.llm import cross_entropy
from cs336_basics.optimizer import get_lr_cosine_schedule, get_lr_wsd_schedule, gradient_clipping
from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint

from .builder import (
    Runtime,
    copy_grads_model_to_optimizer,
    copy_weights_model_to_optimizer,
    copy_weights_optimizer_to_model,
    load_dataset,
)
from .config import RunPaths, TrainConfig
from .metrics import MetricWriter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        runtime: Runtime,
        paths: RunPaths,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        writer: MetricWriter,
    ):
        self.cfg = cfg
        self.rt = runtime
        self.paths = paths
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.train_data = load_dataset(cfg.data.train_path)
        self.valid_data = load_dataset(cfg.data.valid_path)
        self.start_step = self._resume() if cfg.run.resume else 0

    def _resume(self) -> int:
        if not self.paths.checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume; checkpoint does not exist: {self.paths.checkpoint_path}")
        step = load_checkpoint(self.paths.checkpoint_path, self.model, self.optimizer)
        copy_weights_model_to_optimizer(self.model, self.optimizer)
        return step

    def _set_lr(self, adamw_lr: float, muon_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group.setdefault("initial_lr", param_group["lr"])
            param_group["lr"] = muon_lr if param_group.get("use_muon", False) else adamw_lr

    def _scheduled_lr(self, lr_max: float, lr_min: float, step: int) -> float:
        cfg = self.cfg.optim
        if cfg.schedule == "wsd":
            return get_lr_wsd_schedule(
                step,
                lr_max,
                lr_min,
                cfg.warmup_iters,
                cfg.total_iters,
                cfg.wsd_decay_iters(),
            )
        return get_lr_cosine_schedule(step, lr_max, lr_min, cfg.warmup_iters, cfg.total_iters)

    def train_step(self) -> float:
        cfg = self.cfg
        x, y = get_batch(self.train_data, cfg.data.batch_size, cfg.model.context_length, cast(str, self.rt.device))

        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = cross_entropy(logits.float(), y)

        loss.backward()
        copy_grads_model_to_optimizer(self.model, self.optimizer)
        clip_params = [param for group in self.optimizer.param_groups for param in group["params"]]
        gradient_clipping(clip_params, cfg.optim.max_grad_norm, cfg.optim.eps)
        self.optimizer.step()
        copy_weights_optimizer_to_model(self.model, self.optimizer)
        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        cfg = self.cfg
        self.model.eval()
        losses = []
        for _ in range(cfg.run.eval_iters):
            x, y = get_batch(
                self.valid_data,
                cfg.run.eval_batch_size,
                cfg.model.context_length,
                cast(str, self.rt.device),
            )
            logits = self.model(x)
            losses.append(cross_entropy(logits.float(), y).item())
        return sum(losses) / len(losses)

    def run(self) -> None:
        cfg, runtime = self.cfg, self.rt
        on_cuda = runtime.device.type == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(runtime.device)

        run_started = time.perf_counter()
        interval_seconds = 0.0
        interval_steps = 0

        for step in range(self.start_step + 1, cfg.optim.total_iters + 1):
            step_started = time.perf_counter()
            lr = self._scheduled_lr(cfg.optim.learning_rate, cfg.optim.lr_min, step)
            muon_lr = self._scheduled_lr(cfg.optim.muon_learning_rate, cfg.optim.muon_lr_min, step)
            self._set_lr(lr, muon_lr)
            train_loss = self.train_step()

            interval_seconds += time.perf_counter() - step_started
            interval_steps += 1

            valid_loss = None
            if step % cfg.run.eval_interval == 0:
                valid_loss = self.evaluate()
                self.model.train()

            if step % cfg.run.save_interval == 0:
                save_checkpoint(self.model, self.optimizer, step, self.paths.checkpoint_path)

            if step % cfg.run.log_interval == 0 or valid_loss is not None:
                record = {
                    "step": step,
                    "learning_rate": lr,
                    "muon_learning_rate": muon_lr,
                    "train_loss": train_loss,
                    "ms_per_step": 1000 * interval_seconds / interval_steps,
                    "wall_time_seconds": time.perf_counter() - run_started,
                }
                interval_seconds, interval_steps = 0.0, 0
                if on_cuda:
                    record["peak_memory_gb"] = torch.cuda.max_memory_allocated(runtime.device) / 1024**3
                if valid_loss is not None:
                    record["valid_loss"] = valid_loss
                self.writer.write(record)

        if self.start_step < cfg.optim.total_iters and cfg.optim.total_iters % cfg.run.save_interval != 0:
            save_checkpoint(self.model, self.optimizer, cfg.optim.total_iters, self.paths.checkpoint_path)
        logger.info("saved checkpoint: %s", self.paths.checkpoint_path)
        logger.info("saved metrics: %s", self.paths.log_path)
