#!/usr/bin/env python
# coding: utf-8



# In[13]:


import logging
import os

from utils import io_aux_train as training
from utils import config


# #Display original table
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', 100)
# # First que list all objects in the database:
# conn = sqlite3.connect('db.sqlite3')
# pd.read_sql(""" SELECT * FROM sqlite_master;""", conn)
# # load tables:
# summ_vars = pd.read_sql("""SELECT * FROM Indicators;""", conn)
# long = pd.read_sql("""SELECT * FROM CountryIndicators;""", conn)
# 
# # we save the long database with a reshape: year x variables in datasets.txt
# country = long[long['CountryCode'].str.contains("ARB")][["IndicatorCode","Year","Value"]]
# df = country.pivot(index='Year', columns='IndicatorCode',values='Value')
# df = df.fillna(method='ffill') # impute NA with the previous observation.
# df['Country'] = "ARB"
# for n in long.CountryCode.unique()[1:10]: #long.CountryCode.unique()
# #for n in long.CountryCode.unique():
#     country2 = long[long['CountryCode'].str.contains(n)][["IndicatorCode","Year","Value"]]
#     df2 = country2.pivot(index='Year', columns='IndicatorCode',values='Value')
#     df2 = df2.fillna(method='ffill')
#     df2['Country'] = n
#     df = df.append(df2)
# #np.unique(df.Country)
# #df.shape
# #np.savetxt('/home/ferran/Desktop/master/R&I/ml/final_project_skeleton/dataset.txt', result, delimiter='\t',fmt='%s')
# #         header=
# df.columns = df.columns.str.replace(".", "_")
# aa = df.copy()
# 
# aa =range(1,len(df))
# df.shape # 1018
# # Require that many non-NA values.
# df.dropna(thresh=np.quantile(aa, 0.8), axis=1).shape # 238
# df = df.fillna(method='bfill')

# In[16]:


#log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),
log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep),
                       'logs')
log_fname = os.path.join(log_dir, 'io.log')
logging.basicConfig(filename=log_fname, level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s')


# In[8]:


def reduce_dataset(X, year):
    """Reduces the dataset X.
    Removes all the data except the data of the
    X: covariable-cleaned database
    year = (year_of_prediction - 1).

    Parameters
    ----------
    X: covariable-cleaned database
    year: year of prediction
    year_min: Minimum available year in the database.
    Per default: 1960.
    year_max: Maximum available year in the database.
    Per default: 2010.

    Returns
    -------
    red_X
        reduced dataset."""

    drop_list = [i for i in range(config.DB_YEAR_MIN, config.DB_YEAR_MAX + 1)]
    drop_list.remove(year - 1)
    red_X = X.drop(drop_list, axis=0)
    return red_X


# In[9]:


def expand_dataset(X, bst, groups, year):
    """Expands the input data for the model.
    Approximates/predicts the data of the year = (year_of_prediction - 1),
    in case year_of_prediction > config.DB_YEAR_MAX + 2.

    Parameters
    ----------
    X: Data of the last available year (this data can be also a prediction).
    bst: Trained Booster model.
    groups: group indices.
    year: year to predict.

    Note that if, for example, we want to predict the GPD growth of 2012,
    but we only have available data until 2010 from the database, we would
    need to predict the data of 2011 and then use this an an input
    for the final prediction of 2012.

    The prediction of the input data copies the data from the last available
    data from the database and changes the value of the response variable,
    after predicting it.

    Returns
    -------
    X
     predicted data"""
    y_pred = predict(X, bst, groups)
    X[config.PREDICTED_INDICATOR.replace(".","_")] = y_pred
    return X


# In[17]:


def retrieve_test_data(X, bst, year):
    """Retrieve test data. If the year of prediction is greater than
    the database maximum, then it predicts/expands the input data
    for the model for the actual prediction.
    Parameters
    ----------
    X: covariable-cleaned database
    bst: Trained Booster model.
    year: year of prediction

    Returns
    -------
    X_test
        Input test data for the model
    group_test
        group indices of test data"""
    logging.info('Retrieving test dataset')
    if year < (config.DB_YEAR_MIN + 1):
        raise ValueError(f"The year to predict has to be equal                          or greater than {(config.DB_YEAR_MIN + 1)}")
    elif year > 2011:
        first_expand_year = config.DB_YEAR_MAX + 1
        X_test = reduce_dataset(X, config.DB_YEAR_MAX + 1)
        X_test, group_test = training.handle_country_groups(X_test)
        first_expand_year = config.DB_YEAR_MAX + 1
        expand_list = [first_expand_year + i
                       for i in range(year - first_expand_year)]
        for year in expand_list:
            X_test = expand_dataset(X_test, bst, group_test, year)
    else:
        X_test = reduce_dataset(X, year)
        X_test, group_test = training.handle_country_groups(X_test)
    return X_test, group_test


# In[11]:


def predict(X_test, bst, group_test):
    """Gives the response variables for each country.
    Parameters
    ----------
    X_test
        Input test data for the model.
    bst
        Trained Booster model.
    group_test
        group indices of test data.

    Returns
    -------
    y_pred
        Vector of predictions.
    """
    pred = bst.predict(data=X_test, group_data_pred=group_test)
    y_pred = pred['fixed_effect'] + pred['random_effect_mean']
    return y_pred


# (bst, X_train, y_train, data_train,
#  groups_train, gp_model, opt_num_boost_rounds) = training.train()

# #Esto de abajo supongo que tambien va para el analysis del model
# X_test, group_test = retrieve_test_data(2010)
# y_pred = predict(X_test, group_test)
# #X_test["NY_GDP_MKTP_KD_ZG"]
# y_pred

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




