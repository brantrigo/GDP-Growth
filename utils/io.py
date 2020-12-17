# Standard library imports
import sqlite3
import pandas as pd
import numpy as np

# Related third party imports

# Local application/library specific imports
from utils import read_database as rd

def retrieve_training_dataset(database_path, exclude_list):
    """ Retrieves most relevant variables from trainning.
    :param database_path: OS path to database.
    :exclude_list: List of excluded countries or regions.

    """
    # Get values from all countries but those in exclude_list
    long = rd.get_data(database_path, exclude_list)
    # Reshape and prepare Dataframe for linear model. Fit Linear Model and residuals
    df = rd.prepare_data(long)
    # Reject Indicators whose NaN values exceed threshold
    df_fewNA = rd.clean_data(df)
    # Select the top TOP values that better explain GDP Growth
    selected_variables = rd.select_data(df_fewNA, 50)
    
    
    
    
def retrieve_predict_dataset():
    print("Not finished yet")
    pass
