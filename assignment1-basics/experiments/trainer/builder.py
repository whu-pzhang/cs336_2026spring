"""Build torch objects from training configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.llm import TransformerLM
from cs336_basics.optimizer import AdamW, MuonWithAuxAdam

from .config import ModelConfig, OptimConfig, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    device: torch.device
    param_dtype: torch.dtype


def select_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device does not exist: {device}")
    elif requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    elif requested.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    return requested


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.manual_seed(seed)
        np.random.seed(seed)


def resolve_param_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if device.type == "cuda":
        # is_bf16_supported() inspects the default device, not the requested one.
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            raise RuntimeError(f"bf16 needs compute capability >= 8.0; {device} is sm_{major}x")
    elif device.type != "cpu":
        raise ValueError(f"bf16 is not supported on {device.type}")
    return torch.bfloat16


def setup_runtime(run: RunConfig) -> Runtime:
    # Preserve the pre-refactor RNG order: seed before constructing the model.
    set_seed(run.seed)
    device = select_device(run.device)
    param_dtype = resolve_param_dtype(run.dtype, device)
    logger.info("device=%s param_dtype=%s (optimizer state is fp32)", device, param_dtype)
    return Runtime(device=device, param_dtype=param_dtype)


def place_model(
    model: torch.nn.Module,
    runtime: Runtime,
    *,
    torch_compile: bool = False,
) -> torch.nn.Module:
    model = model.to(device=runtime.device, dtype=runtime.param_dtype)
    if torch_compile:
        logger.info("torch.compile enabled")
        model = torch.compile(model)
    return model


def _optimizer_params(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [param for group in optimizer.param_groups for param in group["params"]]


def _paired_params(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model_to_optimizer = getattr(optimizer, "_model_to_optimizer", None)
    if model_to_optimizer is not None:
        return ((param, model_to_optimizer[id(param)]) for param in model.parameters())
    return zip(model.parameters(), _optimizer_params(optimizer), strict=True)


def uses_master_weights(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    first_param = next(model.parameters())
    model_to_optimizer = getattr(optimizer, "_model_to_optimizer", None)
    if model_to_optimizer is not None:
        return model_to_optimizer[id(first_param)] is not first_param
    return first_param is not _optimizer_params(optimizer)[0]


def copy_grads_model_to_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not uses_master_weights(model, optimizer):
        return
    for param, master in _paired_params(model, optimizer):
        master.grad = None if param.grad is None else param.grad.detach().float()


def copy_weights_optimizer_to_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not uses_master_weights(model, optimizer):
        return
    with torch.no_grad():
        for param, master in _paired_params(model, optimizer):
            param.copy_(master)


def copy_weights_model_to_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not uses_master_weights(model, optimizer):
        return
    with torch.no_grad():
        for param, master in _paired_params(model, optimizer):
            master.copy_(param.float())


def build_model(model_cfg: ModelConfig) -> TransformerLM:
    return TransformerLM(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        d_model=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        rope_theta=model_cfg.rope_theta,
        remove_rmsnorm=model_cfg.remove_rmsnorm,
        use_post_norm=model_cfg.use_post_norm,
        norm_eps=model_cfg.norm_eps,
        remove_rope=model_cfg.remove_rope,
        ffn_type=model_cfg.ffn_type,
        qk_norm=model_cfg.qk_norm,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        zero_init_projections=model_cfg.zero_init_projections,
        fused_attention=model_cfg.fused_attention,
    )


def build_optimizer(optim_cfg: OptimConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    model_params = list(model.parameters())
    optimizer_params = [
        param if param.dtype == torch.float32 else param.detach().float().clone().requires_grad_(True)
        for param in model_params
    ]
    model_to_optimizer = {
        id(model_param): optimizer_param for model_param, optimizer_param in zip(model_params, optimizer_params)
    }

    if optim_cfg.optimizer == "adamw":
        optimizer = AdamW(
            optimizer_params,
            lr=optim_cfg.learning_rate,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
    elif optim_cfg.optimizer == "muon":
        muon_params = []
        adam_params = []
        for name, model_param in model.named_parameters():
            optimizer_param = model_to_optimizer[id(model_param)]
            is_hidden_matrix = model_param.ndim >= 2 and "token_embeddings" not in name and "lm_head" not in name
            (muon_params if is_hidden_matrix else adam_params).append(optimizer_param)

        param_groups = [
            {
                "params": adam_params,
                "use_muon": False,
                "lr": optim_cfg.learning_rate,
                "betas": optim_cfg.betas,
                "eps": optim_cfg.eps,
                "weight_decay": optim_cfg.weight_decay,
            },
            {
                "params": muon_params,
                "use_muon": True,
                "lr": optim_cfg.muon_learning_rate,
                "momentum": optim_cfg.muon_momentum,
                "nesterov": optim_cfg.muon_nesterov,
                "ns_steps": optim_cfg.muon_ns_steps,
                "weight_decay": optim_cfg.muon_weight_decay,
            },
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"unsupported optimizer: {optim_cfg.optimizer}")

    # Groups may be reordered by optimizer type; retain an explicit mapping for
    # BF16 master-weight synchronization and gradient copying.
    optimizer._model_to_optimizer = model_to_optimizer
    return optimizer


def load_dataset(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")
