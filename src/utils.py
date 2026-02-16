import logging
import os
import sys

def setup_logging(log_name: str = "experiment", level=logging.INFO):
    # create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")

    # define format
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # configure handler
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]

    # configure root-logger
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=handlers, force=True)

    logging.info(f"Initialized logging {log_path}")