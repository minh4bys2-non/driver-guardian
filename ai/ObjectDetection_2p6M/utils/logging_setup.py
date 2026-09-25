import logging
import os
import sys
from datetime import datetime

from src.config import TrainConfig

FlushFileHandler = logging.FileHandler


def setup_logging(cfg: TrainConfig) -> logging.Logger:
    """Configure the root logger to write one timestamped run log."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_path = os.path.join(
        cfg.log_dir, f"{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    level = getattr(logging, cfg.log_level.upper())
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    handlers = [FlushFileHandler(log_path, encoding="utf-8")]
    if cfg.log_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    logger = logging.getLogger(cfg.run_name)
    logger.info("Logger initialized: %s", log_path)
    return logger
