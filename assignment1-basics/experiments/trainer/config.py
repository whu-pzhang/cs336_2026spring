"""Typed, serializable training configuration."""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


MODEL_CONFIG_SCHEMA_VERSION = 1


@dataclass
class ModelConfig:
    architecture_schema_version: int = MODEL_CONFIG_SCHEMA_VERSION
    num_layers: int = 4
    num_heads: int = 16
    hidden_size: int = 512
    d_ff: int = 1344
    vocab_size: int = 32768
    context_length: int = 256
    rope_theta: float = 10000.0
    remove_rmsnorm: bool = False
    use_post_norm: bool = False
    norm_eps: float = 1e-5
    remove_rope: bool = False
    ffn_type: Literal["swiglu", "silu"] = "swiglu"
    qk_norm: bool = False
    tie_word_embeddings: bool = False
    zero_init_projections: bool = False
    fused_attention: bool = False
    torch_compile: bool = False


@dataclass
class DataConfig:
    train_path: Path
    valid_path: Path
    batch_size: int = 32


@dataclass
class OptimConfig:
    optimizer: Literal["adamw", "muon"] = "adamw"
    learning_rate: float = 1e-3
    lr_min: float = 1e-4
    warmup_iters: int = 400
    total_iters: int = 40000
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    muon_learning_rate: float = 2e-2
    muon_lr_min: float = 2e-3
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    schedule: Literal["cosine", "wsd"] = "cosine"
    decay_frac: float = 0.15

    def wsd_decay_iters(self) -> int:
        return max(1, round(self.total_iters * self.decay_frac))


@dataclass
class RunConfig:
    output_dir: Path
    device: str
    dtype: Literal["fp32", "bf16"] = "fp32"
    seed: int = 42
    resume: bool = False
    save_interval: int = 1000
    eval_interval: int = 1000
    eval_iters: int = 20
    eval_batch_size: int = 16
    log_interval: int = 50
    wandb: bool = False
    wandb_project: str = "cs336-assignment1"
    wandb_run: str | None = None
    wandb_entity: str | None = None


@dataclass
class RunPaths:
    checkpoint_path: Path
    log_path: Path
    config_path: Path


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    run: RunConfig

    def validate(self) -> None:
        positive = {
            "model.num_layers": self.model.num_layers,
            "model.num_heads": self.model.num_heads,
            "model.hidden_size": self.model.hidden_size,
            "model.d_ff": self.model.d_ff,
            "model.vocab_size": self.model.vocab_size,
            "model.context_length": self.model.context_length,
            "model.rope_theta": self.model.rope_theta,
            "model.norm_eps": self.model.norm_eps,
            "data.batch_size": self.data.batch_size,
            "optim.total_iters": self.optim.total_iters,
            "run.eval_batch_size": self.run.eval_batch_size,
            "run.eval_iters": self.run.eval_iters,
            "run.save_interval": self.run.save_interval,
            "run.eval_interval": self.run.eval_interval,
            "run.log_interval": self.run.log_interval,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
        if self.model.architecture_schema_version != MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported model architecture schema version: "
                f"{self.model.architecture_schema_version}; expected {MODEL_CONFIG_SCHEMA_VERSION}"
            )
        if self.model.ffn_type not in {"swiglu", "silu"}:
            raise ValueError("invalid model choices: model.ffn_type")
        if self.model.hidden_size % self.model.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_size = self.model.hidden_size // self.model.num_heads
        if not self.model.remove_rope and head_size % 2 != 0:
            raise ValueError("RoPE requires hidden_size / num_heads to be even")
        if self.optim.learning_rate <= 0 or self.optim.lr_min < 0 or self.optim.learning_rate <= self.optim.lr_min:
            raise ValueError("require learning_rate > lr_min >= 0")
        if self.optim.warmup_iters >= self.optim.total_iters:
            raise ValueError("warmup_iters must be smaller than total_iters")
        if self.optim.optimizer not in {"adamw", "muon"}:
            raise ValueError("optim.optimizer must be either 'adamw' or 'muon'")
        if self.optim.schedule not in {"cosine", "wsd"}:
            raise ValueError("optim.schedule must be either 'cosine' or 'wsd'")
        if self.optim.schedule == "wsd":
            if not 0 < self.optim.decay_frac < 1:
                raise ValueError("optim.decay_frac must satisfy 0 < decay_frac < 1")
            decay_iters = self.optim.wsd_decay_iters()
            if self.optim.warmup_iters + decay_iters >= self.optim.total_iters:
                raise ValueError("WSD requires warmup_iters + decay_iters < total_iters")
        if self.optim.muon_learning_rate <= 0:
            raise ValueError("optim.muon_learning_rate must be positive")
        if self.optim.muon_lr_min < 0 or self.optim.muon_learning_rate <= self.optim.muon_lr_min:
            raise ValueError("require muon_learning_rate > muon_lr_min >= 0")
        if not 0 <= self.optim.muon_momentum < 1:
            raise ValueError("optim.muon_momentum must satisfy 0 <= momentum < 1")
        if self.optim.muon_weight_decay < 0:
            raise ValueError("optim.muon_weight_decay must be non-negative")
        if self.optim.muon_ns_steps <= 0:
            raise ValueError("optim.muon_ns_steps must be positive")
        if not Path(self.data.train_path).exists():
            raise FileNotFoundError(self.data.train_path)
        if not Path(self.data.valid_path).exists():
            raise FileNotFoundError(self.data.valid_path)

    def resolve_paths(self) -> RunPaths:
        output_dir = Path(self.run.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "ckpt.pt"
        log_path = output_dir / "ckpt.jsonl"
        config_path = output_dir / "config.json"
        if self.run.resume:
            if not config_path.is_file():
                raise FileNotFoundError(f"cannot resume; config sidecar does not exist: {config_path}")
            _check_model_match(load_model_config(config_path), self.model, config_path)
        save_config(self, config_path)
        return RunPaths(checkpoint_path=checkpoint_path, log_path=log_path, config_path=config_path)


def _check_model_match(saved: ModelConfig, requested: ModelConfig, path: Path) -> None:
    differing = [
        f"{field.name}: checkpoint={getattr(saved, field.name)!r} requested={getattr(requested, field.name)!r}"
        for field in dataclasses.fields(ModelConfig)
        if getattr(saved, field.name) != getattr(requested, field.name)
    ]
    if differing:
        raise ValueError(
            f"model config in {path} does not match the requested architecture:\n  " + "\n  ".join(differing)
        )


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def config_to_dict(cfg: TrainConfig) -> dict:
    return dataclasses.asdict(cfg, dict_factory=lambda items: {key: _jsonable(value) for key, value in items})


def save_config(cfg: TrainConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(cfg), indent=2) + "\n", encoding="utf-8")


def _normalize_run_dict(raw: dict) -> dict:
    run = dict(raw)
    if "output_dir" not in run:
        if "checkpoint_path" not in run:
            raise ValueError("run config needs output_dir")
        run["output_dir"] = str(Path(run["checkpoint_path"]).parent)
    run.pop("checkpoint_path", None)
    run.pop("log_path", None)
    run["output_dir"] = Path(run["output_dir"])
    return run


def load_config(path: Path) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = dict(raw["data"])
    data["train_path"] = Path(data["train_path"])
    data["valid_path"] = Path(data["valid_path"])
    optim = dict(raw["optim"])
    optim["betas"] = tuple(optim["betas"])
    run = _normalize_run_dict(raw["run"])
    return TrainConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**data),
        optim=OptimConfig(**optim),
        run=RunConfig(**run),
    )


def load_model_config(path: Path) -> ModelConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(**raw["model"])


def parse_train_config(argv: list[str] | None = None) -> TrainConfig:
    if argv is None:
        argv = sys.argv[1:]
    config_path, rest = _split_config_arg(argv)
    kwargs: dict = {"use_underscores": True, "args": rest}
    if config_path is not None:
        kwargs["default"] = load_config(config_path)
    return tyro.cli(TrainConfig, **kwargs)


def _split_config_arg(argv: list[str]) -> tuple[Path | None, list[str]]:
    config_path = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"--config", "-c"}:
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                raise ValueError("--config requires a path")
            config_path = Path(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--config="):
            value = arg.partition("=")[2]
            if not value:
                raise ValueError("--config requires a path")
            config_path = Path(value)
            i += 1
            continue
        rest.append(arg)
        i += 1
    return config_path, rest
