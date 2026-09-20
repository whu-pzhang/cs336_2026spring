import pytest
import torch

from cs336_basics.llm import (
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SiluFFN,
    SwiGLUFFN,
    TransformerBlock,
)
from trainer.builder import (
    Runtime,
    build_model,
    build_optimizer,
    copy_grads_model_to_optimizer,
    copy_weights_model_to_optimizer,
    copy_weights_optimizer_to_model,
    place_model,
    resolve_param_dtype,
    select_device,
)
from trainer.config import ModelConfig, OptimConfig


def tiny_model_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(num_layers=2, num_heads=4, hidden_size=32, d_ff=64, vocab_size=97, context_length=16)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def test_build_model_produces_expected_logits_shape():
    cfg = tiny_model_config()
    model = build_model(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    assert model(tokens).shape == (2, cfg.context_length, cfg.vocab_size)


def test_build_model_maps_hidden_size_to_d_model():
    model = build_model(tiny_model_config())
    assert model.token_embeddings.weight.shape == (97, 32)


def test_build_model_defaults_match_baseline_architecture():
    model = build_model(tiny_model_config())
    block = model.layers[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(model.ln_final, RMSNorm)
    assert block.norm_type == "pre"
    assert isinstance(model.positional_encoder, RotaryPositionalEmbedding)
    assert isinstance(block.attn.positional_encoder, RotaryPositionalEmbedding)
    assert isinstance(block.ffn, SwiGLUFFN)


def test_build_model_shares_rope_across_layers():
    model = build_model(tiny_model_config())
    rope = model.positional_encoder
    assert isinstance(rope, RotaryPositionalEmbedding)
    assert model.layers[0].attn.positional_encoder is rope
    assert model.layers[1].attn.positional_encoder is rope


def test_build_model_no_rms_uses_identity():
    model = build_model(tiny_model_config(remove_rmsnorm=True))
    block = model.layers[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.ln1, torch.nn.Identity)
    assert isinstance(block.ln2, torch.nn.Identity)
    assert isinstance(model.ln_final, torch.nn.Identity)


def test_build_model_post_norm_sets_block_ordering():
    model = build_model(tiny_model_config(use_post_norm=True))
    assert model.layers[0].norm_type == "post"


def test_build_model_uses_configured_norm_eps():
    model = build_model(tiny_model_config(norm_eps=1e-6))
    block = model.layers[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.ln1, RMSNorm)
    assert isinstance(block.ln2, RMSNorm)
    assert isinstance(model.ln_final, RMSNorm)
    assert block.ln1.eps == 1e-6
    assert block.ln2.eps == 1e-6
    assert model.ln_final.eps == 1e-6


def test_build_model_nope_disables_rope():
    model = build_model(tiny_model_config(remove_rope=True))
    assert model.positional_encoder is None
    assert model.layers[0].attn.positional_encoder is None


def test_build_model_ffn_type_and_d_ff_are_explicit():
    silu = build_model(tiny_model_config(ffn_type="silu", d_ff=80))
    assert isinstance(silu.layers[0].ffn, SiluFFN)
    assert silu.layers[0].ffn.w1.weight.shape == (80, 32)
    assert isinstance(build_model(tiny_model_config()).layers[0].ffn, SwiGLUFFN)


def test_build_model_zero_init_projections():
    baseline = build_model(tiny_model_config())
    model = build_model(tiny_model_config(zero_init_projections=True))
    for block in model.layers:
        assert torch.count_nonzero(block.attn.output_proj.weight) == 0
        assert torch.count_nonzero(block.ffn.w2.weight) == 0
        assert torch.count_nonzero(block.attn.q_proj.weight) > 0
        assert torch.count_nonzero(block.ffn.w1.weight) > 0
    assert torch.count_nonzero(model.lm_head.weight) > 0
    assert torch.count_nonzero(baseline.layers[0].attn.output_proj.weight) > 0


def test_build_model_fused_attention_flag():
    baseline = build_model(tiny_model_config())
    model = build_model(tiny_model_config(fused_attention=True))
    assert baseline.layers[0].attn.fused_attention is False
    assert model.layers[0].attn.fused_attention is True


def test_build_optimizer_reads_optim_config():
    model = build_model(tiny_model_config())
    optim_cfg = OptimConfig(learning_rate=3e-4, betas=(0.8, 0.99), eps=1e-7, weight_decay=0.05)
    optimizer = build_optimizer(optim_cfg, model)
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-4
    assert tuple(group["betas"]) == (0.8, 0.99)
    assert group["eps"] == 1e-7
    assert group["weight_decay"] == 0.05


def test_resolve_param_dtype_fp32():
    assert resolve_param_dtype("fp32", torch.device("cpu")) is torch.float32


def test_resolve_param_dtype_bf16_on_cpu():
    assert resolve_param_dtype("bf16", torch.device("cpu")) is torch.bfloat16


def test_place_model_bf16_casts_parameters_and_buffers():
    runtime = Runtime(device=torch.device("cpu"), param_dtype=torch.bfloat16)
    model = place_model(build_model(tiny_model_config()), runtime)
    assert next(model.parameters()).dtype is torch.bfloat16
    assert model.positional_encoder.cos.dtype is torch.bfloat16


def test_build_optimizer_shares_params_for_fp32_model():
    runtime = Runtime(device=torch.device("cpu"), param_dtype=torch.float32)
    model = place_model(build_model(tiny_model_config()), runtime)
    optimizer = build_optimizer(OptimConfig(), model)
    assert optimizer.param_groups[0]["params"][0] is next(model.parameters())


def test_build_optimizer_uses_fp32_masters_for_bf16_model():
    runtime = Runtime(device=torch.device("cpu"), param_dtype=torch.bfloat16)
    model = place_model(build_model(tiny_model_config()), runtime)
    optimizer = build_optimizer(OptimConfig(), model)
    master = optimizer.param_groups[0]["params"][0]
    param = next(model.parameters())
    assert param.dtype is torch.bfloat16
    assert master.dtype is torch.float32
    assert master is not param


def test_bf16_optimizer_step_keeps_fp32_state_and_updates_model():
    runtime = Runtime(device=torch.device("cpu"), param_dtype=torch.bfloat16)
    model = place_model(build_model(tiny_model_config()), runtime)
    optimizer = build_optimizer(OptimConfig(learning_rate=1e-2, weight_decay=0.0), model)
    param = next(model.parameters())
    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    copy_grads_model_to_optimizer(model, optimizer)
    optimizer.step()
    copy_weights_optimizer_to_model(model, optimizer)
    master = optimizer.param_groups[0]["params"][0]
    state = optimizer.state[master]
    assert param.dtype is torch.bfloat16
    assert state["m"].dtype is torch.float32
    assert state["v"].dtype is torch.float32
    assert not torch.equal(param, before)
    torch.testing.assert_close(param.float(), master.detach(), atol=2e-2, rtol=2e-2)


def test_copy_model_weights_to_optimizer_syncs_masters():
    runtime = Runtime(device=torch.device("cpu"), param_dtype=torch.bfloat16)
    model = place_model(build_model(tiny_model_config()), runtime)
    optimizer = build_optimizer(OptimConfig(), model)
    param = next(model.parameters())
    with torch.no_grad():
        param.fill_(0.5)
    copy_weights_model_to_optimizer(model, optimizer)
    master = optimizer.param_groups[0]["params"][0]
    torch.testing.assert_close(master, torch.full_like(master, 0.5))


def test_select_device_rejects_unsupported_type():
    with pytest.raises(ValueError, match="unsupported device"):
        select_device("meta")
