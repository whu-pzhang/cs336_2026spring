"""Write run metrics to JSONL, W&B, and the logger."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import TrainConfig, config_to_dict

logger = logging.getLogger(__name__)


class MetricWriter:
    def __init__(self, cfg: TrainConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = Path(log_path)
        self.wandb_run = None

    def __enter__(self) -> MetricWriter:
        if self.cfg.run.wandb:
            import wandb

            run_name = self.cfg.run.wandb_run or Path(self.cfg.run.output_dir).name
            if self.cfg.run.wandb_entity:
                self.wandb_run = wandb.init(
                    project=self.cfg.run.wandb_project,
                    name=run_name,
                    config=config_to_dict(self.cfg),
                    entity=self.cfg.run.wandb_entity,
                )
            else:
                self.wandb_run = wandb.init(
                    project=self.cfg.run.wandb_project,
                    name=run_name,
                    config=config_to_dict(self.cfg),
                )
            logger.info("wandb run: %s", self.wandb_run.url)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return False

    def write(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log({key: value for key, value in record.items() if key != "step"}, step=record["step"])
        logger.info("%s", self.format_record(record))

    @staticmethod
    def format_record(record: dict) -> str:
        parts = [
            f"step={record['step']}",
            f"train_loss={record['train_loss']:.6f}",
            f"lr={record['learning_rate']:.6g}",
            f"ms/step={record['ms_per_step']:.1f}",
        ]
        if "peak_memory_gb" in record:
            parts.append(f"peak_mem={record['peak_memory_gb']:.2f}GiB")
        if "valid_loss" in record:
            parts.append(f"valid_loss={record['valid_loss']:.6f}")
        return " ".join(parts)
