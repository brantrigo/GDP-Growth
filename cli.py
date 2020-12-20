#!/usr/bin/env python
import os
import logging
import argparse
import gpboost as gpb
from datetime import datetime

from utils import config, io, models
from utils import io_aux_train as training
from utils import io_aux_test as testing

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
    	print("test")
    	logging.info("Determining relevant covariables")
    	X  = io.retrieve_clean_dataset(database_path=config.DATABASE_PATH, exclude_list=config.exclude_list, PREDICTED_INDICATOR=config.PREDICTED_INDICATOR)
    	logging.info('Starting the train')
    	X.columns = X.columns.str.replace(".","_")
    	print(X.columns)
    	X = X.reset_index(drop=True)
    	pred_ind  = config.PREDICTED_INDICATOR.replace(".","_")
    	(X_train, y_train, data_train, groups_train) = training.retrieve_training_dataset(X, 
    	predicted_indicator=pred_ind)
    	(gp_model, params, opt_num_boost_rounds) = training.get_booster_model(data_train, groups_train)
    	bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=opt_num_boost_rounds)
    	logging.info('Starting the prediction')
    	X_test, group_test = testing.retrieve_test_data(X, bst, 2012)
    	print(X_train.columns)
    	print(X_test.columns)
    	pred = bst.predict(data=X_test, group_data_pred=group_test)
    	y_pred = pred['fixed_effect'] + pred['random_effect_mean']
    	print(y_pred)
    	
