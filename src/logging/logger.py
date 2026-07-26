import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logger(name: str = "LogiEdge",
               log_file: str = "logs/logiedge.log",
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_048_576,
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
