#!/usr/bin/env python
# coding: utf-8

# In[10]:


import gpboost as gpb
import numpy as np
import pandas as pd
import logging
import os

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

# In[2]:



# In[12]:


#log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),
log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep),                       
                       'logs')
log_fname = os.path.join(log_dir, 'io.log')
logging.basicConfig(filename=log_fname, level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s')


# In[4]:


def handle_country_groups(X):
    """Generates a group index list with all the countries.
    This indices will be used by the model to know which entries of
    the database belong to the same group, in this case, which entries
    belong to the same country.
    Gets rid of the 'Country' column, the model does not accept non-
    numerical values.
    Parameters
    ----------
    X: Database

    Returns
    -------
    X_wo_countries
        Database without the country column
    groups
        List with the indices of each group
        """
    groups = X[["Country"]].replace(pd.unique(X.Country),
                                    list(range(0, len(pd.unique(X.Country)))))
    groups = np.concatenate(groups.to_numpy())
    #X_wo_countries = X.drop("Country", axis=1)
    X_wo_countries = X.drop("Country", axis=1)
    return X_wo_countries, groups


# In[3]:


def prepare_training_dataset(X, y):
    """Deletes all rows that have the highest year index in X (here, 2010).
    Delete all rows that have the lowest index in y (here, 1960).
    This is done to prepare the model for predicting y_{t+1}.
    The country groups are handled here, too (see train.handle_country_groups).
    Parameters
    ----------
    X: Database data
    y: Response data

    Returns
    -------
    X_train
        Train data
    y_train
        Response train data
    groups_train
        Group indices"""
    print(X)
    print(y)
    X = X.drop(config.DB_YEAR_MAX, axis=0)
    y_train = y.drop(config.DB_YEAR_MIN, axis=0)
    #X = X.drop(index = config.DB_YEAR_MIN - 1 , level=1)
    #X = X.drop(index = config.DB_YEAR_MAX, level=1)
    #y_train = y.drop(index = config.DB_YEAR_MIN, level=1)
    #X = X[X.Time != config.DB_YEAR_MAX]
    #y = y[y.Time != config.DB_YEAR_MIN]
    X_train, groups_train = handle_country_groups(X)
    return X_train, y_train, groups_train


# In[2]:


def retrieve_training_dataset(X, predicted_indicator):
    """
    Transforms the raw dataset into a dataset that is suitable
    for training the model.
    Parameters
    ----------
    X: Covariable-cleaned database

    Returns
    -------
    X_train
        Train data
    y_train
        Response train data
    data_train
        Train data readable for the package gpbooster, contains the information
        about X_train and y_train
    groups_train
        Group indices
        """
    logging.info('Retrieving training dataset')
    y = X[[predicted_indicator]]
    X_train, y_train, groups_train = prepare_training_dataset(X, y)
    data_train = gpb.Dataset(X_train, y_train)
    return X_train, y_train, data_train, groups_train


# In[7]:


def get_booster_model(data_train, groups_train):
    """Gets model and define its parameters. For finding the optimal number
    of iterations, cross-validation is applied.

    Parameters
    ----------
    data_train: Train data readable for the package gpbooster,
    should contain the information about X_train and y_train
    groups_train: Group indices

    Returns
    -------
    gp_model
        Instance of the Gradient Tree boosting model with random effects
    params
        Parameters with which the model should be trained
    opt_num_boost_rounds
        Optimal number of boosting rounds for the training, found with cross-
        validation
        """
    logging.info('Getting booster model')
    gp_model = gpb.GPModel(group_data=groups_train)
    gp_model.set_optim_params(params={"optimizer_cov": "fisher_scoring"})
    params = {
        'objective': 'regression_l2',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_data_in_leaf': 5,
        'verbose': 0
    }
    logging.info('Calculating optimal number of boost rounds                  via cross-validation')
    cvbst = gpb.cv(params=params, train_set=data_train,
                   gp_model=gp_model, use_gp_model_for_validation=True,
                   num_boost_round=300, early_stopping_rounds=5,
                   nfold=3, verbose_eval=False, show_stdv=False, seed=1)
    opt_num_boost_rounds = np.argmin(cvbst['l2-mean'])
    return gp_model, params, opt_num_boost_rounds


# In[8]:


def train(X):
    """Trains the Gradient tree boosting model with random effects.
    It automatically reads and processes the data from the database.
    Parameters
    ----------
    X: Covariable-cleaned database

    Returns
    -------
    bst
        The trained Booster model.
    X_train
        Train data
    y_train
        Response train data
    data_train
        Train data readable for the package gpbooster, contains the information
        about X_train and y_train
    groups_train
        Group indices
    gp_model
        Instance of the Gradient Tree boosting model with random effects
    opt_num_boost_rounds
        Optimal number of boosting rounds for the training, found with cross-
        validation
    """
    logging.info('Starting the train')
    X_train, y_train, data_train, groups_train = retrieve_training_dataset(X)
    gp_model, params, opt_num_boost_rounds = get_booster_model(data_train,
                                                               groups_train)
    bst = gpb.train(params=params, train_set=data_train,
                    gp_model=gp_model, num_boost_round=opt_num_boost_rounds)
    return (bst, X_train, y_train, data_train, groups_train,
            gp_model, opt_num_boost_rounds)


# X = df
# (bst, X_train, y_train, data_train,
#  groups_train, gp_model, opt_num_boost_rounds) = train()

# In[ ]:




