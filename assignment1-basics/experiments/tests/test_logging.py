import logging

from train import configure_logging


def test_configure_logging_prints_llm_info(capsys):
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    try:
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        configure_logging()
        logging.getLogger("cs336_basics.llm").info("number of non-embedding parameters: 0.01M")
        captured = capsys.readouterr()
        assert "number of non-embedding parameters" in captured.err + captured.out
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
