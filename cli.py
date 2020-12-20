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
    choices=["predict"],
    help="Task to be performed",
)

parser.add_argument(
    "--year", 
    nargs="?",
    help="Year of prediction",
    default=2011
)
    

if __name__ == "__main__":
    args = parser.parse_args()
    if args.task == "predict":
        logging.info("Determining relevant covariables")
        #data = io.retrieve_clean_dataset(config.DATABASE_PATH, config.exclude_list, config.PREDICTED_INDICATOR)
        #print(data)
        print(args.year)
        logging.info("Training")
    if args.task == "predict":
        logging.info("Predicting")
