import logging
import os
import sys
from pathlib import Path


def setup_logger(name, save_dir, filename='training.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(Path(save_dir) / filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
