import pandas as pd
import numpy as np
import sqlite3
import statsmodels.formula.api as smf
from sklearn.feature_selection import f_regression, mutual_info_regression
import os.path   
from utils import config


def get_data(database_path, exclude_list = None):
    """Retrieve data from the database, excluding names in exclude_list"""
    conn = sqlite3.connect(database_path)
    country_names = pd.read_sql("""SELECT * FROM Countries;""", conn)
    not_country = country_names.loc[country_names["LongName"].isin(exclude_list)]["CountryCode"]
    long = pd.read_sql("""SELECT * FROM CountryIndicators;""", conn)
    long = long.loc[~long["CountryCode"].isin(not_country)]
    return long

def prepare_data(long, PREDICTED_INDICATOR):
    """Create a Multi-index Dataframe and prepare data for linear model"""
    df = long.pivot_table(index=['CountryCode','Year'], columns='IndicatorCode',values='Value',aggfunc=np.sum)
    # Create 3 more columns with Countries, Objective Indicator lag and year
    df['Country'] = df.index.get_level_values(0)
    df['lag1'] = df[PREDICTED_INDICATOR].shift(1)
    df['Time'] = df.index.get_level_values(1)
    # Extract Rows where Predicted Indicator and its lag do not have values
    df = df.dropna(subset=[PREDICTED_INDICATOR,"lag1"])
    # Countries strings to numeric values
    groups = df[["Country"]].replace(pd.unique(df.Country), 
            list(range(0,len(pd.unique(df.Country)))))
    groups = pd.to_numeric(groups.Country)
    return df, groups

def linear_model(df1, PREDICTED_INDICATOR, groups):
    # Replace . by _ for linear model
    df1.columns = df1.columns.str.replace(".", "_")
    predicted_indicator = PREDICTED_INDICATOR.replace(".", "_")
    # Mixed linear model with group as random effect.
    df1_sub = df1[[predicted_indicator,"Country","lag1"]]
    string = f"{predicted_indicator} ~ lag1"
    md = smf.mixedlm(string, df1_sub, groups=groups)
    mod = md.fit()
    print(mod.summary())
    df1['residuals'] = mod.resid
    df1['Country'] = groups
    mod.summary()
    return df1

def clean_data(df, threshold = 0.3):
    """Reject Indicators whose NaN values exceed threshold """
    # Filter/impute vars with NA
    df_fewNA = df[df.columns[(df.isnull().sum(axis=0)/df.shape[0]<=threshold)]]
    country2 = df_fewNA['Country']
    df_fewNA = df_fewNA.groupby(country2).transform(lambda x: x.fillna(x.ffill().bfill()))
    df_fewNA = df_fewNA.fillna(df.mean())
    df_fewNA["Country"] = country2
    return df_fewNA

def select_data(df_fewNA, num_features = 50):
    # Feature selection
    covs = df_fewNA.drop(["NY_GDP_MKTP_KD_ZG","residuals"], 1)
    Y = df_fewNA[['residuals']]
    info = mutual_info_regression(covs, np.ravel(Y))
    df_varimp =pd.DataFrame(data={'name': covs.columns, 'varimp': info})
    # Keep top50
    selected_variables = df_varimp.sort_values(by="varimp",ascending=False)[0:49]
    selected_variables['name'] = selected_variables['name'].str.replace('_','.')
    selected_variables['name'].to_csv(path_or_buf='./utils/selected_variables.txt',header=True, 
                                          index=None, sep='\t', mode='a')
    return selected_variables


def get_select_data(database_path,exclude_list, PREDICTED_INDICATOR, file='./utils/selected_variables.txt'):
    if os.path.isfile(file):
        selected_variables= pd.read_csv(file)   
        conn = sqlite3.connect(database_path)
        selected_variables = selected_variables['name'].append(pd.Series("NY.GDP.MKTP.KD.ZG"),ignore_index=True).tolist()
        queryString = 'SELECT * FROM CountryIndicators WHERE IndicatorCode IN (\'{}\');'.format('\',\''.join([_ for _ in selected_variables]))
        vars1 = pd.read_sql(queryString, con=conn)
        # LongName to CountryName
        country_names = pd.read_sql("""SELECT LongName,CountryCode FROM Countries;""", conn)
        not_country = country_names.loc[country_names["LongName"].isin(exclude_list)]["CountryCode"]
        vars1= vars1.loc[~vars1["CountryCode"].isin(not_country)]
        vars2 = vars1.pivot_table(index=['CountryCode','Year'], columns='IndicatorCode',values='Value',aggfunc=np.sum)
        vars2['Country'] = vars2.index.get_level_values(0)
        vars2['Time'] = vars2.index.get_level_values(1)
        vars2['lag1'] = vars2[PREDICTED_INDICATOR].shift(1)
        return vars2










