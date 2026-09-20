import json
from pathlib import Path

from trainer.config import DataConfig, ModelConfig, OptimConfig, RunConfig, TrainConfig
from trainer.metrics import MetricWriter


def make_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(),
        data=DataConfig(train_path=tmp_path / "t.npy", valid_path=tmp_path / "v.npy"),
        optim=OptimConfig(),
        run=RunConfig(output_dir=tmp_path, device="cpu"),
    )


def test_write_appends_one_json_object_per_call(tmp_path):
    log_path = tmp_path / "m.jsonl"
    with MetricWriter(make_cfg(tmp_path), log_path) as writer:
        writer.write({"step": 1, "learning_rate": 1e-3, "train_loss": 2.0, "ms_per_step": 5.0})
        writer.write({"step": 2, "learning_rate": 9e-4, "train_loss": 1.5, "ms_per_step": 5.0})
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[0]["train_loss"] == 2.0


def test_write_preserves_key_order(tmp_path):
    log_path = tmp_path / "m.jsonl"
    record = {
        "step": 1,
        "learning_rate": 1e-3,
        "train_loss": 2.0,
        "ms_per_step": 5.0,
        "wall_time_seconds": 0.5,
        "peak_memory_gb": 1.25,
        "valid_loss": 2.5,
    }
    with MetricWriter(make_cfg(tmp_path), log_path) as writer:
        writer.write(record)
    assert list(json.loads(log_path.read_text()).keys()) == list(record.keys())


def test_format_record_matches_pre_refactor_string():
    record = {
        "step": 50,
        "learning_rate": 0.000125,
        "train_loss": 8.744955,
        "ms_per_step": 102.34,
        "wall_time_seconds": 12.0,
        "peak_memory_gb": 10.4375,
        "valid_loss": 5.146352,
    }
    assert MetricWriter.format_record(record) == (
        "step=50 train_loss=8.744955 lr=0.000125 ms/step=102.3 peak_mem=10.44GiB valid_loss=5.146352"
    )


def test_format_record_omits_optional_fields():
    record = {"step": 50, "learning_rate": 0.001, "train_loss": 8.0, "ms_per_step": 100.0}
    assert MetricWriter.format_record(record) == "step=50 train_loss=8.000000 lr=0.001 ms/step=100.0"


def test_writer_without_wandb_does_not_import_it(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.run.wandb is False
    with MetricWriter(cfg, tmp_path / "m.jsonl") as writer:
        assert writer.wandb_run is None
