import logging
import os
import sys
from pathlib import Path

def setup_logger(name, save_dir, filename="training.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent double logging if re-initialized

    # Check if handlers already exist
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = Path(save_dir) / filename
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
