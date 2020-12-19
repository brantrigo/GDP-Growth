#!/usr/bin/env python
import os
import logging
import argparse
from datetime import datetime

from utils import config, io, models


logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, datetime.now().strftime('cli_%Y-%m-%d_%H:%M:%S.log')),
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "task",
    choices=["train", "predict"],
    help="Task to be performed",
)
# You can add here custom optional arguments to your program

# database_path = config.DATABASE_PATH

if __name__ == "__main__":
    args = parser.parse_args()
    if args.task == "train":
        data = io.retrieve_clean_dataset(config.DATABASE_PATH, config.exclude_list, config.PREDICTED_INDICATOR)
        print(data)
        logging.info("Training")
    if args.task == "predict":
        logging.info("Predicting")
