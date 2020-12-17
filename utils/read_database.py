import pandas as pd
import numpy as np
import sqlite3
import statsmodels.formula.api as smf
from sklearn.feature_selection import f_regression, mutual_info_regression

def get_data(database_path, exclude_list = None):
    """Retrieve data from the database, excluding names in exclude_list"""
    conn = sqlite3.connect(database_path)
    country_names = pd.read_sql("""SELECT * FROM Countries;""", conn)
    not_country = country_names.loc[country_names["LongName"].isin(exclude_list)]["CountryCode"]
    long = pd.read_sql("""SELECT * FROM CountryIndicators;""", conn)
    long = long.loc[~long["CountryCode"].isin(not_country)]
    return long

def prepare_data(long):
    """Create a Multi-index Dataframe and prepare data for linear model"""
    # Reshape dataframe
    df = long.pivot_table(index=['CountryCode','Year'], columns='IndicatorCode',values='Value',aggfunc=np.sum)
    # Substitue _ by . for compatibility with statsmodels.formula.api
    df['Country'] = df.index.get_level_values(0)
    df.columns = df.columns.str.replace(".", "_")
    # Creation of covariables for the linear model
    df['lag1'] = df['NY_GDP_MKTP_KD_ZG'].shift(1)
    df['Time'] = df.index.get_level_values(1)
    df = df.dropna(subset=["NY_GDP_MKTP_KD_ZG","lag1"])
    # Countries strings to numeric values:
    groups = df[["Country"]].replace(pd.unique(df.Country), 
            list(range(0,len(pd.unique(df.Country)))))
    groups = pd.to_numeric(groups.Country)
    # Mixed linear model with group as random effect.
    df_sub = df[["NY_GDP_MKTP_KD_ZG","Country","lag1"]]
    md = smf.mixedlm("NY_GDP_MKTP_KD_ZG ~ lag1", df_sub, groups=groups)#re_formula="~Time",groups=groups,  #df["Country"].to_numpy()
    mod = md.fit()
    print(mod.summary())
    df['residuals'] = mod.resid
    df['Country'] = groups
    mod.summary()
    return df

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
    return selected_variables




    