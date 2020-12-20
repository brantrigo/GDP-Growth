# Standard library imports
import sqlite3
import pandas as pd
import numpy as np

# Related third party imports

# Local application/library specific imports
from utils import read_database as rd

def retrieve_clean_dataset(database_path, exclude_list, PREDICTED_INDICATOR):
    """ Retrieves most relevant variables from trainning.
    :param database_path: OS path to database.
    :exclude_list: List of excluded countries or regions.

    """
    # Get values from all countries but those in exclude_list
    long = rd.get_data(database_path, exclude_list)
    # Reshape and prepare Dataframe
    df, groups = rd.prepare_data(long, PREDICTED_INDICATOR)
    # Fit Linear Model and get residuals
    df1 = df.copy()
    df1 = rd.linear_model(df1, PREDICTED_INDICATOR, groups)
    # Reject Indicators whose NaN values exceed threshold
    df_fewNA = rd.clean_data(df1)
    # Select the top TOP values that better explain GDP Growth
    selected_variables = rd.select_data(df_fewNA, 50)
    # Convert countries to numeric values and extract the desired columns from dataframe
    df['Country'] = groups
    lst = selected_variables['name'].to_list()
    data = df[lst]
    print(data['Country'])
    return data
    
    
def retrieve_predict_dataset():
    print("Not finished yet")
    pass
