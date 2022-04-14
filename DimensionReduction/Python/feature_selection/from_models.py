import numpy as np, pandas as pd, seaborn as sns, ml.dataframe.datagrid as dg
import os
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, SelectKBest
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from common_functions import append_dict, partition, df_reorder_columns, timestamp, require_tuple

from dataframe.datasets import DataSet, to_DataSet, hold_dataset
from run_batch import column_filter, run_batch
__author__ = "Clinten Graham"
__maintainer__ = "Clinten Graham"  # 2/20
__email__ = "clintengraham@gmail.com"
__status__ = "Prototype"


def plot_coefs(df, feature_cnames, target_cname, models = [LinearRegression()], plot = True):
    '''
    :param models: List of estimators with coef_ attribute.
      LinearRegression(), Ridge(), Lasso()
    '''
    df = df[[target_cname] + feature_cnames].dropna()  # prescaled
    X = df[feature_cnames].to_numpy()
    y = df[target_cname].to_numpy()

    results_df = pd.DataFrame()
    results_df['Features'] = feature_cnames
    results_df.set_index('Features', inplace=True)
    models = models if isinstance(models, list) else [models]
    model_cnames = []
    for mdl in models:
        mdl = mdl.fit(X, y)
        results_df[f'{type(mdl).__name__}'] = mdl.coef_
        model_cnames += [f'{type(mdl).__name__}']
    if len(models) > 1:
        results_df['avg'] = results_df[model_cnames].mean(axis=1)
        results_df = results_df.iloc[results_df['avg'].abs().argsort()[::-1]]
        del results_df['avg']
    if plot:
        results_df.plot(kind='bar')
        plt.title(f'Feature Coefficients for {target_cname}')
        plt.xlabel('Features')
        plt.ylabel('Coefficients')
        plt.show()
    print(results_df)
# estimators = all_estimators()
#
# for name, class_ in estimators:
#     if hasattr(class_, 'coef_'):
#         print(name)