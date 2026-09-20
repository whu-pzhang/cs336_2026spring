import json
from pathlib import Path

import pytest

from trainer.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    load_config,
    load_model_config,
    parse_train_config,
    save_config,
)


def make_cfg(tmp_path: Path, **overrides) -> TrainConfig:
    train_path = tmp_path / "train.npy"
    valid_path = tmp_path / "valid.npy"
    train_path.write_bytes(b"")
    valid_path.write_bytes(b"")
    cfg = TrainConfig(
        model=ModelConfig(),
        data=DataConfig(train_path=train_path, valid_path=valid_path),
        optim=OptimConfig(),
        run=RunConfig(output_dir=tmp_path / "run", device="cpu"),
    )
    for dotted, value in overrides.items():
        group, name = dotted.split(".")
        setattr(getattr(cfg, group), name, value)
    return cfg


def test_validate_accepts_defaults(tmp_path):
    make_cfg(tmp_path).validate()


@pytest.mark.parametrize(
    "dotted",
    [
        "model.num_layers",
        "model.num_heads",
        "model.hidden_size",
        "model.d_ff",
        "model.vocab_size",
        "model.context_length",
        "model.rope_theta",
        "model.norm_eps",
        "data.batch_size",
        "optim.total_iters",
        "run.eval_batch_size",
        "run.eval_iters",
        "run.save_interval",
        "run.eval_interval",
        "run.log_interval",
    ],
)
def test_validate_rejects_non_positive(tmp_path, dotted):
    cfg = make_cfg(tmp_path, **{dotted: 0})
    with pytest.raises(ValueError, match="must be positive"):
        cfg.validate()


def test_validate_rejects_indivisible_hidden_size(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.hidden_size": 513})
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate()


def test_validate_rejects_unsupported_model_schema(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.architecture_schema_version": 2})
    with pytest.raises(ValueError, match="schema version"):
        cfg.validate()


def test_validate_rejects_invalid_ffn_type(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.ffn_type": "invalid"})
    with pytest.raises(ValueError, match="invalid model choices"):
        cfg.validate()


def test_validate_rejects_odd_rope_head_size(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.hidden_size": 36, "model.num_heads": 4})
    with pytest.raises(ValueError, match="RoPE"):
        cfg.validate()


def test_validate_allows_odd_head_size_without_rope(tmp_path):
    make_cfg(
        tmp_path,
        **{
            "model.hidden_size": 36,
            "model.num_heads": 4,
            "model.remove_rope": True,
        },
    ).validate()


def test_validate_rejects_lr_min_not_below_lr(tmp_path):
    cfg = make_cfg(tmp_path, **{"optim.lr_min": 1e-3})
    with pytest.raises(ValueError, match="learning_rate > lr_min"):
        cfg.validate()


def test_validate_rejects_warmup_not_below_total(tmp_path):
    cfg = make_cfg(tmp_path, **{"optim.warmup_iters": 40000})
    with pytest.raises(ValueError, match="warmup_iters"):
        cfg.validate()


def test_validate_allows_zero_warmup(tmp_path):
    make_cfg(tmp_path, **{"optim.warmup_iters": 0}).validate()


def test_validate_rejects_invalid_schedule(tmp_path):
    cfg = make_cfg(tmp_path, **{"optim.schedule": "linear"})
    with pytest.raises(ValueError, match="optim.schedule"):
        cfg.validate()


def test_validate_rejects_wsd_without_stable_phase(tmp_path):
    cfg = make_cfg(
        tmp_path,
        **{
            "optim.schedule": "wsd",
            "optim.total_iters": 100,
            "optim.warmup_iters": 40,
            "optim.decay_frac": 0.7,
        },
    )
    with pytest.raises(ValueError, match="warmup_iters \\+ decay_iters"):
        cfg.validate()


def test_validate_accepts_wsd(tmp_path):
    make_cfg(tmp_path, **{"optim.schedule": "wsd", "optim.decay_frac": 0.15}).validate()


def test_validate_rejects_missing_train_path(tmp_path):
    cfg = make_cfg(tmp_path, **{"data.train_path": tmp_path / "nope.npy"})
    with pytest.raises(FileNotFoundError):
        cfg.validate()


def test_resolve_paths_writes_ckpt_log_and_config_in_output_dir(tmp_path):
    cfg = make_cfg(tmp_path)
    paths = cfg.resolve_paths()
    assert paths.checkpoint_path == tmp_path / "run" / "ckpt.pt"
    assert paths.log_path == tmp_path / "run" / "ckpt.jsonl"
    assert paths.config_path == tmp_path / "run" / "config.json"
    assert paths.config_path.exists()
    assert paths.checkpoint_path.parent.is_dir()


def test_save_load_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.remove_rope": True, "optim.betas": (0.8, 0.99)})
    path = tmp_path / "config.json"
    save_config(cfg, path)
    assert cfg == load_config(path)


def test_load_config_accepts_legacy_checkpoint_path(tmp_path):
    cfg = make_cfg(tmp_path)
    path = tmp_path / "config.json"
    save_config(cfg, path)
    raw = json.loads(path.read_text())
    raw["run"]["checkpoint_path"] = str(Path(raw["run"].pop("output_dir")) / "ckpt.pt")
    raw["run"]["log_path"] = str(Path(raw["run"]["checkpoint_path"]).with_suffix(".jsonl"))
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.run.output_dir == tmp_path / "run"


def test_saved_config_is_json_with_string_paths(tmp_path):
    cfg = make_cfg(tmp_path)
    path = tmp_path / "config.json"
    save_config(cfg, path)
    raw = json.loads(path.read_text())
    assert raw["model"]["num_layers"] == 4
    assert raw["model"]["architecture_schema_version"] == 1
    assert raw["model"]["remove_rmsnorm"] is False
    assert raw["model"]["use_post_norm"] is False
    assert raw["model"]["rope_theta"] == 10000.0
    assert raw["model"]["remove_rope"] is False
    assert raw["model"]["ffn_type"] == "swiglu"
    assert raw["model"]["qk_norm"] is False
    assert raw["model"]["tie_word_embeddings"] is False
    assert raw["model"]["zero_init_projections"] is False
    assert raw["model"]["fused_attention"] is False
    assert raw["model"]["torch_compile"] is False
    assert raw["optim"]["schedule"] == "cosine"
    assert raw["optim"]["decay_frac"] == 0.15
    assert "ablation" not in raw["model"]
    assert isinstance(raw["data"]["train_path"], str)
    assert Path(raw["run"]["output_dir"]).name == "run"
    assert "checkpoint_path" not in raw["run"]
    assert "log_path" not in raw["run"]


def test_load_model_config_reads_only_model_section(tmp_path):
    cfg = make_cfg(tmp_path, **{"model.vocab_size": 10000})
    path = tmp_path / "config.json"
    save_config(cfg, path)
    assert load_model_config(path) == cfg.model


def test_resume_with_matching_architecture_is_allowed(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.resolve_paths()
    cfg.run.resume = True
    cfg.resolve_paths()


def test_resume_without_sidecar_fails(tmp_path):
    cfg = make_cfg(tmp_path)
    paths = cfg.resolve_paths()
    paths.config_path.unlink()
    cfg.run.resume = True
    with pytest.raises(FileNotFoundError, match="config sidecar"):
        cfg.resolve_paths()
    assert not paths.config_path.exists()


def test_resume_with_mismatched_architecture_lists_fields(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.resolve_paths()
    cfg.run.resume = True
    cfg.model.num_layers = 8
    cfg.model.vocab_size = 999
    cfg.model.remove_rope = True
    with pytest.raises(ValueError) as excinfo:
        cfg.resolve_paths()
    message = str(excinfo.value)
    assert "num_layers" in message and "vocab_size" in message and "remove_rope" in message
    assert "num_heads" not in message


def test_parse_train_config_loads_file(tmp_path):
    cfg = make_cfg(tmp_path, **{"optim.total_iters": 123})
    path = tmp_path / "recipe.json"
    save_config(cfg, path)
    loaded = parse_train_config(["--config", str(path)])
    assert loaded.optim.total_iters == 123
    assert loaded.run.device == "cpu"
    assert loaded.run.output_dir == tmp_path / "run"


def test_parse_train_config_cli_overrides_file(tmp_path):
    path = tmp_path / "recipe.json"
    save_config(make_cfg(tmp_path), path)
    loaded = parse_train_config(["--config", str(path), "--optim.total_iters", "99", "--run.dtype", "bf16"])
    assert loaded.optim.total_iters == 99
    assert loaded.run.dtype == "bf16"


def test_parse_train_config_requires_config_path():
    with pytest.raises(ValueError, match="--config"):
        parse_train_config(["--config"])
