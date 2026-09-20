"""Train a Transformer LM; implementation lives in experiments/trainer."""

import logging

from trainer import builder
from trainer.config import parse_train_config
from trainer.loop import Trainer
from trainer.metrics import MetricWriter


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    cfg = parse_train_config()
    cfg.validate()
    paths = cfg.resolve_paths()

    runtime = builder.setup_runtime(cfg.run)
    model = builder.place_model(
        builder.build_model(cfg.model),
        runtime,
        torch_compile=cfg.model.torch_compile,
    )
    optimizer = builder.build_optimizer(cfg.optim, model)

    with MetricWriter(cfg, paths.log_path) as writer:
        Trainer(cfg, runtime, paths, model, optimizer, writer).run()


if __name__ == "__main__":
    main()
