import numpy as np, pandas as pd, seaborn as sns, dataframe.datagrid as dg
import os
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, SelectKBest
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from common_functions import append_dict, partition, df_reorder_columns, timestamp, require_tuple

from dataframe.datasets import DataSet, to_DataSet, hold_dataset, require_extracted
from data.ShapeUp.ShapeUp_data import *
from run_batch import column_filter, run_batch
__author__ = "Clinten Graham"
__maintainer__ = "Clinten Graham"  # 5/20
__email__ = "clintengraham@gmail.com"
__status__ = "Prototype"


def plot_correlation(
        df: pd.DataFrame,
        feature_cnames: list,
        target_cnames: list,
        fontsize=8
    ) -> None:
    '''
    Plot correlation matrix given features and targets, right now there is no distinction between the two, may start
    highlighting target names.

    :param df: dataframe
    :param feature_cnames: list of column names
    :param target_cnames: list of column names
    :param fontsize: Fontsize for labels
    :return: None. Image printed with plt.show()
    '''
    df = df[feature_cnames+target_cnames]
    # Remove Sex from column list, it does not appear in matrix
    if 'DEM_SEX' in df.columns:
        # df['DEM_SEX'].map({'M': 0, 'F': 1}) # did not work with df.corr
        del df['DEM_SEX']

    # Draw heatmap of df
    corr_df = df.dropna().corr()  # make correlation matrix
    g = sns.heatmap(
        corr_df,
        annot=len(feature_cnames+target_cnames) < 10,
        annot_kws={"fontsize":fontsize, "rotation":-45},
        cbar_kws={"orientation": "vertical"}
    )

    # Style options
    g.set_aspect("equal")
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)

    g.set_yticklabels(df.columns, fontsize=fontsize, rotation=0)
    # g.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # x_ticks = list(df.columns)
    # g.set_xticks(np.arange(len(x_ticks)))
    # g.set_xticklabels(x_ticks, fontsize=fontsize, horizontalalignment='center')
    plt.show()
    return corr_df


def feature_explore(df, feature_cname, age_groups=False) -> pd.DataFrame:
    """
    Print histograms of columns in feature_cname and return dataframe of statistics on each column.

    :param df: pandas dataframe
    :param feature_cname: Column name from df (or list of column names)
    :param age_groups: Bool. Describe dataframe over age groups
    :return: dataframe of statistics on each feature_cname
    """
    if isinstance(feature_cname, str):
        feature_cname = [feature_cname]
    # df.hist(by='DEM_SEX',
    #         # sharex=True,
    #         )
    dfm = df.loc[df['DEM_SEX'] == 'M']
    dfm = dfm[feature_cname + ['DEM_AGE']]
    dff = df.loc[df['DEM_SEX'] == 'F']
    dff = dff[feature_cname + ['DEM_AGE']]

    n_feat = len(feature_cname)
    if n_feat == 1:
        # df.hist(column=feature_cname,
        #         by='DEM_SEX',
        #         # sharex=True,
        #         histtype='barstacked',
        #         label=feature_cname
        #         )
        plt.hist([dfm[feature_cname[0]].dropna(), dff[feature_cname[0]].dropna()],
                 histtype='barstacked')
        plt.show()
    else:
        # if n_feat < 6:
        #     f, axes = plt.subplots(2, n_feat, figsize=(3 * 3, 3 * n_feat / 2))
        # else:
        #     # # More rectangular plots for large feature sets
        #     # n_col = math.ceil(n_feat / math.ceil(math.sqrt(len(feature_cname))))
        #     # f, axes = plt.subplots(n_col, 2*n_col, figsize=(6*n_col, 3*n_col))
        n_row = 2*((n_feat // 5) + 1)
        n_col = min(5, n_feat)
        f, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

        for i, feature in enumerate(feature_cname):
            k = 2 * i
            sns.distplot(dfm[feature].dropna(), color="skyblue", ax=axes[k % n_row, k // n_row])
            k = k + 1
            sns.distplot(dff[feature].dropna(), color="pink", ax=axes[k % n_row, k // n_row])
        plt.show()

    if age_groups:
        # describe by age group if option is True
        age_groups = {**{"18 to 29": [18, 29]},
                      **{str(n) + " to " + str(n + 9): [n, n + 9] for n in range(30, 61, 10)},
                      **{"70 to 90": [70, 90]}}
        dfm_list = []
        dff_list = []
        for group_name, group_range in age_groups.items():
            dfm_list += [dfm.loc[(group_range[0] <= dfm['DEM_AGE']) & (dfm['DEM_AGE'] <= group_range[1])].describe()]
            dff_list += [dff.loc[(group_range[0] <= dff['DEM_AGE']) & (dff['DEM_AGE'] <= group_range[1])].describe()]
        dfm = pd.concat(dfm_list, keys=age_groups.keys())
        dff = pd.concat(dff_list, keys=age_groups.keys())
        df_data = pd.concat([dfm, dff], keys=['M', 'F'])
    else:
        df_data = pd.concat([dfm.describe(), dff.describe()], keys=['M','F'])

    del df_data['DEM_AGE']
    return df_data


def feature_explore_stats(df, feature_cname, age_groups=False) -> pd.DataFrame:
    # feature_explore (stratifies sex and age) but only return feature statistic dataframe (no histograms)
    if isinstance(feature_cname, str):
        feature_cname = [feature_cname]

    dfm = df.loc[df['DEM_SEX'] == 'M']
    dfm = dfm[feature_cname + ['DEM_AGE']]
    dff = df.loc[df['DEM_SEX'] == 'F']
    dff = dff[feature_cname + ['DEM_AGE']]

    if age_groups:
        # describe by age group if option is True
        age_groups = {**{"18 to 29": [18, 29]},
                      **{str(n) + " to " + str(n + 9): [n, n + 9] for n in range(30, 61, 10)},
                      **{"70 to 90": [70, 90]}}
        dfm_list = []
        dff_list = []
        for group_name, group_range in age_groups.items():
            dfm_list += [dfm.loc[(group_range[0] <= dfm['DEM_AGE']) & (dfm['DEM_AGE'] <= group_range[1])].describe()]
            dff_list += [dff.loc[(group_range[0] <= dff['DEM_AGE']) & (dff['DEM_AGE'] <= group_range[1])].describe()]
        dfm = pd.concat(dfm_list, keys=age_groups.keys())
        dff = pd.concat(dff_list, keys=age_groups.keys())
        df_data = pd.concat([dfm, dff], keys=['M', 'F'])
    else:
        df_data = pd.concat([dfm.describe(), dff.describe()], keys=['M','F'])

    del df_data['DEM_AGE']
    return df_data


stats_dict_default = {
    "count": lambda df: np.sum(~np.isnan(df)),
    "mean": np.nanmean,
    "std": np.nanstd,
    "var": np.nanvar,
    "min": np.nanmin,
    "25%": lambda df: np.nanquantile(df, q=.25),
    "50%": lambda df: np.nanquantile(df, q=.5),
    "75%": lambda df: np.nanquantile(df, q=.75),
    "max": np.nanmax,
}
def RefValueTable(df, stats_dict=None) -> pd.DataFrame:
    '''
    Print dataframe of statistics on each column of df from custom dictionary of statistical functions.
    No stratification is performed on df.

    :param df: pd.DataFrame
    :param stats_dict: dictionary with {"func_name": func} key value pairs
    :return: pd.DataFrame
    '''
    if stats_dict is None:
        stats_dict = stats_dict_default
    df = df._get_numeric_data()
    stats_df = pd.DataFrame()
    for stat_name, stat_func in stats_dict.items():
        stats_df[stat_name] = df.apply(stat_func, axis=0)
    return stats_df.T
# print(RefValueTable(df[m_common+m_manual]))
# print(df[m_common+m_manual].describe())  # default describe funtion from pandas


'''
Returns Modified df
'''


def cut_by_IQR(df, q=2, target_cnames = ['BC_DXA_FAT_TOT', 'BC_DXA_LST_TOT']):
    out_list = []
    for target_cname in target_cnames:
        series = df[target_cname].dropna()
        g = sns.boxplot(series)
        plt.show()
        arr = series.to_numpy()
        Q1 = np.quantile(arr, .25)
        Q3 = np.quantile(arr, .75)
        IQR = Q3 - Q1
        BoxMax = Q3 + q * IQR
        outliers = df[['SubjectID', target_cname]].where(series > BoxMax).dropna()
        print(f'Q3 + {q}*IQR = {BoxMax} ({len(outliers)} subjects out of {len(series)})')
        print(outliers['SubjectID'].to_list())
        print(outliers)
        out_list += outliers['SubjectID'].to_list()
    mask = df['SubjectID'].map(lambda x: not(x in out_list))
    return df.loc[mask]


'''
Examples
'''

def corr_cluster(corr_df, thresh=0.5, distance_method='dissimilarity', linkage_method='average', cluster_method='distance'):
    # https://stackoverflow.com/questions/38070478/how-to-do-clustering-using-the-matrix-of-correlation-coefficients
    X = corr_df
    dist = {'pdist': lambda X: sch.distance.pdist(X.values),
            'dissimilarity': lambda X: squareform(1 - np.abs(X))
    }[distance_method]

    hierarchy = sch.linkage(dist(corr_df), method=linkage_method)
    labels = sch.fcluster(hierarchy, thresh * dist(corr_df).max(), cluster_method)
    grouped_columns = [[X.columns.tolist()[j] for j in np.argwhere(labels == i).T[0]] for i in np.unique(labels)]
    return grouped_columns

def feature_explore_examples():
    df_path = 'C:\\Users\\Clint\\PycharmProjects\\BigDataEnergy\\ml\\data\\ShapeUp\\pdDataStorage_v4.xlsx'
    df_original = pd.read_excel(df_path)

    df_ext = require_extracted(df_path, all_features+all_targets)
    corr_df = plot_correlation(df_ext, all_features+['DEM_SEX'], all_targets)

    X = corr_df.values
    d = sch.distance.pdist(X)  # vector of (n choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
    del df_ext['SubjectID']
    del df_ext['SubjectID.1']
    del df_ext['DEM_SEX']
    # plot_correlation(df_ext, list(corr_cluster(df_ext)), all_targets)
    # corr_df = plot_correlation(df_ext, columns, all_targets)

    dissimilarity = 1 - np.abs(corr_df)
    hierarchy = sch.linkage(squareform(1 - np.abs(corr_df)), method='average')
    labels = sch.fcluster(hierarchy, 0.5, criterion='distance')
    print()


    # feature_explore(df_original, 'DA_3DO3_CIRC_Ch')
    # df_data = feature_explore(df_original, m_common+a_b_all, age_groups=True)
    # print(df_data)
    #
    # print(RefValueTable(df_original[m_common+m_manual]))
    # print(df_original[m_common+m_manual].describe())  # default describe funtion from pandas


if __name__ == "__main__":
    feature_explore_examples()