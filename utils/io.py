# Standard library imports
import sqlite3
import pandas as pd
import numpy as np

# Related third party imports
# Local application/library specific imports

def retrieve_training_dataset(database_path):
    """ Retrieves most relevant variables from trainning.
    :param database_path: OS path to database.

    """
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 100) 
    conn = sqlite3.connect(database_path)
    long = pd.read_sql("""SELECT * FROM CountryIndicators;""", conn)
    table = pd.pivot_table(long, values=['Value'],
                       index=['Year'],
                       columns=['IndicatorCode'],
                       aggfunc=np.sum)
    table = table.dropna(axis=0, how="any", thresh=50, subset=None, inplace=False)
    #table.fillna(method='ffill')
    #table.drop(('Value', 1960), axis = 1, inplace=True)
    print(table.iloc[0:9,0:9])
    print(table.shape)
    #table.drop(columns=('Value', 1960))
    
def retrieve_predict_dataset():
    print("Not finished yet")
    pass
