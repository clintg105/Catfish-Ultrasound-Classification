import numpy as np, pandas as pd, seaborn as sns, dataframe.datagrid as dg
import os
#from sklearn.utils.testing import all_estimators
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

'''
Mathematical Functions
'''


def info_entropy(c_binned):
    c_PMF = c_binned / float(np.sum(c_binned))  # normalized
    c_PMF = c_PMF[np.nonzero(c_PMF)]  # for speed, and reduces to 1D case
    return -np.dot(c_PMF, np.log2(c_PMF))

def vect_MI(x, y, bins):
    # WIP: not agreeing w/ fast_vect_MI
    x_bins = np.histogram(x, bins)[0]
    y_bins = np.histogram(y, bins)[0]
    xy_bins = np.histogram2d(x, y, bins)[0]

    Hx = info_entropy(x_bins)
    Hy = info_entropy(y_bins)
    Hxy = info_entropy(xy_bins)
    return Hx + Hy - Hxy

def fast_vect_MI(x, y, bins):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
    xy_bins = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=xy_bins)

def mtrx_MI(m):
    m = m.T  # loop over columns of  m (features) instead of rows (subjects)
    return np.array([[fast_vect_MI(i_feat, j_feat, 5) for j_feat in m] for i_feat in m]) # best number of bins?


'''
Max Relevance, Min Redundancy (Information-Based Feature Selection)
'''


def mRMR(df, feature_cnames, target_cname, metric='covariance', return_best=1,
         best_heatmap=False, best_pairplot=False):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI
    }[metric]
    # the first column is the target, all columns are standardized
    df = df[[target_cname] + feature_cnames].dropna()  # prescaled
    df_cov = mtrx_method(df.to_numpy())
    n = len(feature_cnames)
    # create list of boolean sequences each starting with 1 (see range)
    bool_list = [np.array([b == '1' for b in list('{:0{}b}'.format(i, n))]) \
                 for i in range(2**n + 1, 2**(n + 1))]

    mRMR = np.array([[-np.inf, [], 0, 0] for _ in range(return_best)])
    for mask in bool_list:
        s = np.count_nonzero(mask) - 1  # number of (non-target) features
        covs = df_cov[mask][:, mask]  # apply mask to rows, then columns
        relevance = np.sum(covs[1:, 0]) / s  # target_covs = covs[1:, 0]
        redundancy = np.sum(covs[1:, 1:]) / (s ** 2)  # feature_covs = covs[1:, 1:]
        mRMR0 = relevance - redundancy
        try:
            # check for existence of the first index where mRMR0 is higher than the listed score
            idx = np.nonzero(mRMR[:, 0] < mRMR0)[0][0]
            # all higher indices must be moved up
            if idx < return_best - 1:
                for i in range(idx + 1, return_best)[::-1]:  # preform this operation from the top down to avoid over-writing
                    mRMR[i] = mRMR[i - 1]
            # place mRMR0 in the first slot of mRMR for which mRMR[:, 0] < mRMR0
            mRMR[idx] = [mRMR0, np.array(feature_cnames)[mask[1:]], relevance, redundancy]
        except IndexError:
            continue
    if best_heatmap:
        ax = sns.heatmap(df_cov, xticklabels=[target_cname] + feature_cnames, yticklabels=[target_cname] + feature_cnames)
        for labx, laby in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            text = labx.get_text()
            if text in mRMR[0, 1]:  # lets highlight the best picked features
                labx.set_weight('bold')
                laby.set_weight('bold')
        plt.show()
    if best_pairplot:
        df_pairplot = df[[target_cname] + mRMR[0, 1].tolist()]
        sns.pairplot(df_pairplot, vars=df_pairplot.columns[:-1], markers="+", kind="reg")
        plt.show()
    return np.array(mRMR)


def build_redundancy_list(df, initial_features, feature_pool=[]):
    '''
    Given some list of :initial_features and a :feature_pool
    :returns list with the nth element chosen to be the least redundant with the n-1 previous elements.
    '''
    if not feature_pool:  # not [] == True
        feature_pool = list(set(df.columns) - set(initial_features))
    else:
        for feat in initial_features:
            if feat in feature_pool:
                feature_pool.remove(feat)
    df = df[initial_features + feature_pool].dropna()  # prescaled
    selected_features = initial_features.copy()
    for k in range(len(feature_pool)):
        s = len(initial_features) + (k + 1)  # current number of features
        redundancies = dict()
        for feat in feature_pool:
            df0 = df[[feat] + selected_features]
            # default metric is covariance
            df0_cov = np.cov(df0.to_numpy(), rowvar=False)
            redundancies[feat] = np.sum(df0_cov) / (s ** 2)
        selection = min(redundancies, key=redundancies.get)
        selected_features += [selection]
        feature_pool.remove(selection)
        if not feature_pool:  # feature_pool == []
            return selected_features


# df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean())/x.std(), axis=0).to_numpy()
# print(mtrx_MI(df),'\n', '\n',
#       np.array([[fast_vect_MI(i_feat, j_feat, 5) for j_feat in df.T] for i_feat in df.T])
#       , '\n', '\n') = 5

# df = df[[target_cname] + feature_cnames].dropna()
# X = df[feature_cnames].to_numpy()
# y = df[target_cname].to_numpy()
# f_dict = {feature_cnames[idx]: np.array2string(X[0:4, idx]) for idx in range(len(feature_cnames))}
# f_dict_inv = dict(map(reversed, f_dict.items()))
# print(f_dict, dict(f_dict))
#
# X_ksel = SelectKBest(f_regression, k=k).fit_transform(X, y) # chi2 -ValueError: Input X must be non-negative.
# X_cnames = [f_dict_inv[np.array2string(X[0:4, idx])] for idx in range(k)]
# print(X_ksel)
# print(X_cnames)


def select_n_best_metric(df, feature_cnames, target_cname, n=3, metric='covariance'):
    ''' :returns list of :n features from :feature_cnames with most covariance to :target_cname '''
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI
    }[metric]
    df = df[[target_cname] + feature_cnames].dropna().to_numpy()
    df_cov = mtrx_method(df)
    target_covs = df_cov[0, 1:]
    cov_indices = np.argsort(abs(target_covs))[::-1]  # find indices of target covs, highest to lowest
    selections = [feature_cnames[i] for i in cov_indices[0:n]]
    return selections
def remove_n_worst_metric(df, feature_cnames, target_cname, n=3, metric='covariance'):
    return select_n_best_metric(df, feature_cnames, target_cname, len(feature_cnames) - n, metric)


def select_best_threshold(df, feature_cnames, target_cname, t=.8, metric='covariance'):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI
    }[metric]
    df = df[[target_cname] + feature_cnames].dropna().to_numpy()
    df_cov = mtrx_method(df)
    target_covs = df_cov[0, 1:]
    # target_covs = target_covs/np.sum(target_covs)  # NORMALIZE
    cov_indices = np.argwhere(np.abs(target_covs) > t).flatten()
    return np.array(feature_cnames)[cov_indices]


def lists_from_firsts(list):
    n_max = len(list)
    output_list = []
    for n in range(n_max):
        output_list += [list[:(n + 1)]]
    assert len(list) == len(output_list)
    return output_list


def select_and_run(df, feature_cnames, target_cname, model, upto=10, graph_r2=True):
    dataset = to_DataSet(df)
    # selections = lists_from_firsts(select_n_best_metric(df, feature_cnames, target_cname, n=upto))
    selections = lists_from_firsts(build_redundancy_list(df, ["waist circ", "hip circ"], feature_pool=feature_cnames))
    # selections = [arr.tolist() for arr in mRMR(df, feature_cnames, target_cname, return_best=upto)[:, 1]]
    data_config_dict = {
        'target_cnames': [target_cname],
        'feature_options': {
            'Selected': {len(cname_list): cname_list for cname_list in selections}
        },
        'transform_options': {
            'SEX': {
                "M": [column_filter("SEX", 1)],
                "F": [column_filter("SEX", 0)],
                # "M/F": []
            }
        },
        'scalar_config': {}
    }
    results = run_batch(dataset, data_config_dict, model, n_cores=1)
    if graph_r2:
        name = ['split', 'score_type']
        data_to_col = 'score'
        collapse = (('test_r2', ['test', 'r2']), ('train_r2', ['train', 'r2']))
        dfs = []
        for col in collapse:
            df0 = results.copy()
            df0[data_to_col] = df0[col[0]]
            df0[name[0]] = col[1][0]
            df0[name[1]] = col[1][1]
            dfs += [df0]
        results_plot = pd.concat(dfs)

        # results_plot['Selection Number'] = results_plot.index.map(lambda x: x % upto)
        results_plot['Selection'] = results_plot.index.map(lambda x: selections[-1][x % upto])
        plot = sns.catplot(x='Selection', y="score", hue="split", row='score_type', col="SEX", data=results_plot,
                           kind="bar", aspect=.7)
        plot.set_xticklabels(rotation=85, horizontalalignment='right', fontsize='x-small')
        plt.show()
    pd.set_option('display.max_colwidth', -1)
    # print(results['Selected'].head(upto))
    return results


def select_and_run_targs(df, feature_cnames, targets, model, upto=10, graph_r2=True, **run_batch_kwargs):
    upto_max = upto if isinstance(upto, int) else max(upto)
    dataset = to_DataSet(df)
    tdf = pd.DataFrame()
    score='accuracy'
    for target in targets:
        selections = lists_from_firsts(select_n_best_metric(df, feature_cnames, target, n=upto_max))
        selections = selections if isinstance(upto, int) else [selections[i-1] for i in upto]
        print('[STATUS]', target, 'slections:',selections[-1])

        data_config_dict = {
            'target_cnames': [target],
            'feature_options': {
                'Selected': {str(len(cname_list)): cname_list for cname_list in selections}
            },
            'transform_options': {
                # 'SEX': {
                #     "M": [column_filter("SEX", "M")],
                #     "F": [column_filter("SEX", "F")],
                #     # "M": [column_filter("SEX", 1)],
                #     # "F": [column_filter("SEX", 0)],
                #     # "M/F": []
                # }
            },
            'scalar_config': {}
        }
        results = run_batch(dataset, data_config_dict, model, **run_batch_kwargs)
        tdf = pd.concat([tdf, results])
        if graph_r2:
            # results_plot['Selection Number'] = results_plot.index.map(lambda x: x % upto)
            # results_plot['Selection'] = results_plot.index.map(lambda x: selections[-1][x % upto])
            # plot = sns.catplot(x='Selection', y="mean_test_r2", hue="split", row='target', col="SEX", data=tdf,
            #                    kind="bar", aspect=.7)
            tdf_plot = results.copy()
            if isinstance(upto, int):
                tdf_plot['Selected'] = tdf_plot['Selected'].map(lambda x: selections[-1][int(x.split(', ')[-1])-1])
            else:
                tdf_plot['Selected'] = tdf_plot['Selected'].map(lambda x: int(x))

            sns.set_style("whitegrid")
            sns.despine(left=True)
            g = sns.FacetGrid(tdf_plot, col="Model", row='Target', sharex=False, sharey=True, aspect=1.5)

            g.map(plt.errorbar, "Selected", "avg_test_"+score, "std_test_"+score, marker=".")
            g.map(plt.errorbar, "Selected", "avg_train_"+score, "std_train_"+score, marker=".", color='orange')
            for ax in g.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
            # plot.set_xticklabels(rotation=85, horizontalalignment='right', fontsize='x-small')
            plt.show()
        # print(results['Selected'].head(upto))
    return tdf


def select_and_run_targs_MF(df, feature_cnames, targets, model, upto=10, graph_r2=True, sexes=['M', 'F']):
    n_splits = 5
    tdf = pd.DataFrame()
    for sex in sexes:
        if 'sex' != 'all':
            df0 = df.loc[df['DEM_SEX'] == sex]
        else:
            df0 = df.copy()
        dataset = to_DataSet(df0)
        for target in targets:
            selections = lists_from_firsts(select_n_best_metric(df0, feature_cnames, target, n=len(feature_cnames)))
            data_config_dict = {
                'target_cnames': [target],
                'feature_options': {
                'Selected': {', '.join(cname_list): cname_list for cname_list in selections}
                },
                'transform_options': {
                    'DEM_SEX': {}
                },
                'scalar_config': {
                    # target: StandardScaler
                }
            }
            data_config_dict['transform_options']['DEM_SEX'] = {str(sex): []}
            results = run_batch(dataset, data_config_dict, model, n_cores=1, cv_params=dict(
                n_splits=n_splits,
                scorers=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
                total_train=True, return_params=True
            ))
            tdf = pd.concat([tdf, results])
    tdf = tdf.loc[tdf.index < upto]
    if graph_r2:
        tdf_plot = tdf.copy()

        tdf_plot['Selected'] = tdf_plot['Selected'].map(lambda x: x.split(', ')[-1])
        tdf_plot['var_test_r2'] = tdf_plot['std_test_r2'].map(lambda x: (x**2))
        tdf_plot['var_train_r2'] = tdf_plot['std_train_r2'].map(lambda x: (x**2))
        tdf_plot['se_test_r2'] = tdf_plot.apply(lambda x: x['std_test_r2']/np.sqrt(n_splits), axis=1)
        tdf_plot['se_train_r2'] = tdf_plot.apply(lambda x: x['std_train_r2']/np.sqrt(n_splits), axis=1)
        tdf_plot['error_test'] = tdf_plot.apply(lambda x: [x['avg_test_r2']-min(x['all_test_r2']), max(x['all_test_r2'])-x['avg_test_r2']], axis=1)
        tdf_plot['error_train'] = tdf_plot.apply(lambda x: [x['avg_train_r2']-min(x['all_train_r2']), max(x['all_train_r2'])-x['avg_train_r2']], axis=1)

        tdf_plot = tdf_plot.applymap(lambda strg: strg.replace('DEM_','').replace('DA_','').replace('3DO3_','').replace('BC_','').replace('DXA_','') \
                                    if isinstance(strg, str) else strg)
        g = sns.FacetGrid(tdf_plot, col="Target", row='DEM_SEX', sharex=False, sharey=False)
        g.map(plt.errorbar, "Selected", "avg_test_r2", "std_test_r2", marker=".")
        g.map(plt.errorbar, "Selected", "avg_train_r2", "std_train_r2", marker=".", color='orange')
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
        plt.show()
    pd.set_option('display.max_colwidth', -1)
    return tdf


def select_targs_MF_SIMA(df, feature_cnames, targets=['TOTAL_FAT','TOTAL_LEAN'], sexes=['M', 'F']):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI  # WIP
    }['covariance']
    arr = np.empty((0, len(feature_cnames)))
    r_names = []
    for sex in sexes:
        df0 = df.loc[df['DEM_SEX'] == sex]
        for target in targets:
            r_names += [target + ' (' + sex + ')']
            df1 = df0[[target] + feature_cnames].dropna().to_numpy()
            df_cov = mtrx_method(df1)
            target_covs = df_cov[0, 1:]
            arr = np.vstack([arr, target_covs])
    return pd.DataFrame(data=arr, index=r_names, columns=feature_cnames)


def select_and_run_batches(df, feature_options, target_cname, model, selector_options, upto=10, graph_r2=True):
    dataset = to_DataSet(df)
    data_config_dict = dict()
    data_config_dict['target_cnames'] = [target_cname]
    data_config_dict['feature_options'] = dict()
    for group_name, feature_group in feature_options.items():
        data_config_dict['feature_options'][group_name] = dict()
        data_config_dict['feature_options'][group_name]['all'] = feature_group
        for selector_name, selector in selector_options.items():
            data_config_dict['feature_options'][group_name][selector_name] = \
                selector(df, feature_group, target_cname)
    data_config_dict['scalar_config'] = {}
    data_config_dict['transform_options'] = {'DEM_SEX': {
                "M": [column_filter("DEM_SEX", 1)],
                "F": [column_filter("DEM_SEX", 0)],
                # "M/F": []
    }}
    results = run_batch(dataset, data_config_dict, model, n_cores=1, show_best_runs=3)

    cname_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config_dict['feature_options']))
    cname_dg = cname_dg[(cname_dg['__data__'].map(len) != 0)]  # Remove rows with no input columns
    cname_dg = pd.concat([cname_dg, cname_dg, cname_dg]).reset_index(drop=True)
    results['Features'] = cname_dg['__data__']
    return results


def select_and_nonsequ_MFA(df, feature_options, targets, model, selector_options, selection_groups='all', ignored_groups=[], sexes=['M', 'F']):
    if selection_groups == 'all':
        selection_groups = feature_options.keys()
    n_splits = 5
    tdf = pd.DataFrame()
    for sex in sexes:
        if sex != 'all':
            df0 = df.loc[df['DEM_SEX'] == sex]
        else:
            df0 = df
        dataset = to_DataSet(df0)
        for target in targets:
            print(sex, target)
            data_config_dict = {
                'target_cnames': [target],
                'feature_options': {},
                'transform_options': {
                    'DEM_SEX': {}
                },
                'scalar_config': {
                    # target: StandardScaler
                }
            }
            for group_name, feature_group in feature_options.items():
                data_config_dict['feature_options'][group_name] = dict()
                if group_name in ignored_groups:
                    for k, v in feature_group.items():
                        data_config_dict['feature_options'][group_name][k] = v
                    continue
                data_config_dict['feature_options'][group_name]['none'] = []
                if group_name in selection_groups:
                    for selector_name, selector in selector_options.items():
                        data_config_dict['feature_options'][group_name][selector_name] = \
                            selector(df0, feature_group, target)
                data_config_dict['feature_options'][group_name]['all'] = feature_group
            data_config_dict['transform_options']['DEM_SEX'] = {str(sex): []}
            results = run_batch(dataset, data_config_dict, model, n_cores=-1, cv_params=dict(
                n_splits=n_splits,
                scorers=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
                total_train=True, #return_params=True
            ))

            cname_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config_dict['feature_options']))
            cname_dg = cname_dg[(cname_dg['__data__'].map(len) != 0)]  # Remove rows with no input columns
            cname_dg = pd.concat([cname_dg]).reset_index(drop=True)
            results['#Features'] = cname_dg['__data__'].map(len)
            results['Features'] = cname_dg['__data__']

            tdf = pd.concat([tdf, results])
    pd.set_option('display.max_colwidth', -1)
    return tdf

def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.width', 1000)

    eth = ['DEM_RACE']
    bmi = ['CA_BMI']
    sex = ['DEM_SEX']
    age = ['DEM_AGE']
    # volumes = ['VOL_3DO3_TOT', 'VOL_3DO3_Arm_R', 'VOL_3DO3_Arm_L', 'VOL_3DO3_Leg_R', 'VOL_3DO3_Leg_L', 'VOL_3DO3_Trunk']
    # m_common = ['DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th_R', 'DA_3DO3_CIRC_B_R']
    # m_all = ['DA_3DO3_CIRC_Ch', 'DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th_R', 'DA_3DO3_CIRC_Th_L', 'DA_3DO3_CIRC_C_R', 'DA_3DO3_CIRC_C_L', 'DA_3DO3_CIRC_Wr_R', 'DA_3DO3_CIRC_Wr_L', 'DA_3DO3_CIRC_F_R', 'DA_3DO3_CIRC_F_L', 'DA_3DO3_CIRC_B_R', 'DA_3DO3_CIRC_B_L', 'DA_3DO3_CIRC_A_R', 'DA_3DO3_CIRC_A_L', 'DA_3DO3_LEN_Arm_L', 'DA_3DO3_LEN_Arm_R', 'DA_3DO3_LEN_Leg_L', 'DA_3DO3_LEN_Leg_R']
    # # a_b_common = ["waist circ A_B", "hip circ A_B", 'ThighGirth A_B', 'BicepGirth A_B']
    # a_b_all = ['DA_3DO3_ER_Ch', 'DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th_R', 'DA_3DO3_ER_Th_L', 'DA_3DO3_ER_C_R', 'DA_3DO3_ER_C_L', 'DA_3DO3_ER_Wr_L', 'DA_3DO3_ER_F_R', 'DA_3DO3_ER_F_L', 'DA_3DO3_ER_B_R', 'DA_3DO3_ER_B_L', 'DA_3DO3_ER_A_R', 'DA_3DO3_ER_Wr_R', 'DA_3DO3_ER_A_L']

    volumes = ['VOL_3DO3_TOT', 'VOL_3DO3_Arm', 'VOL_3DO3_Leg', 'VOL_3DO3_Trunk']
    m_common = ['DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th', 'DA_3DO3_CIRC_B']
    m_all = ['DA_3DO3_CIRC_Ch', 'DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th', 'DA_3DO3_CIRC_C',
             'DA_3DO3_CIRC_Wr', 'DA_3DO3_CIRC_F', 'DA_3DO3_CIRC_B', 'DA_3DO3_CIRC_A', 'DA_3DO3_LEN_Arm',
             'DA_3DO3_LEN_Leg']
    a_b_common = ['DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_Ch', 'DA_3DO3_ER_C']
    a_b_all = ['DA_3DO3_ER_Ch', 'DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_C', 'DA_3DO3_ER_Wr',
              'DA_3DO3_ER_F', 'DA_3DO3_ER_B', 'DA_3DO3_ER_A']
    SA_all = ['DA_3DO3_SA_TOT', 'DA_3DO3_SA_Trunk', 'DA_3DO3_SA_Arm_R', 'DA_3DO3_SA_Leg_R']
    m_manual = ['CA_CIRC_Th_R', 'CA_CIRC_H', 'CA_CIRC_W', 'CA_CIRC_B_R']
    all_features = bmi + volumes + SA_all + m_manual + m_all + a_b_all + age + sex
    all_targets = ['BC_DXA_FAT_TOT',
                   'BC_DXA_LST_TOT'
                   ]
    scalar_config = {
        "DEM_SEX": LabelBinarizer,
        "DEM_AGE": MinMaxScaler,
        "default": StandardScaler
    }
    feature_cnames = m_common + volumes
    target_cname = 'BC_DXA_FAT_TOT'

    # model = LinearRegression(normalize=True)
    model = MLPRegressor(solver="lbfgs",
            activation="identity",
            max_iter=800,
            hidden_layer_sizes=(1,))

    df = pd.read_excel('../data/ShapeUp/pdDataStorage_v4.xlsx')
    # df = df.groupby(['SubjectID'], as_index=False).aggregate('mean')
    df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
    df = to_DataSet(df).extract_data(all_features + all_targets, 'BC_DXA_BMD_Leg', scaler_config=scalar_config, data_transformers=[]).x_scaled
    df['SubjectID'] = df.index
    df.to_csv('ext_df_1.csv')
    # df.plot.scatter(x='DA_3DO3_LEN_Leg', y='DA_3DO3_LEN_Arm', c='DEM_SEX', colormap='viridis')
    plt.show()
    df = pd.read_csv('ext_df_1.csv')


    # print(mRMR(df, feature_cnames, target_cname, return_best=10))
    # mRMR(df, feature_cnames, target_cname, return_best=10, best_heatmap=True, best_pairplot=True)
    # print(mRMR(df, feature_cnames, target_cname, method='mutual information', return_best=6))
    # print(build_redundancy_list(df, ["waist circ", "hip circ"], feature_pool=all_features))

    # plot_coefs(df, feature_cnames, target_cname, models = [LinearRegression(), Ridge(), Lasso(alpha=.001)])
    #
    # print(select_n_best_metric(df, feature_cnames, target_cname, n=2))
    # print(remove_n_worst_metric(df, feature_cnames, target_cname, n=2))
    # print(select_best_threshold(df, feature_cnames, target_cname, t=.7))

    # print(select_and_run(df, all_features, target_cname, model, upto=len(all_features)))

    feature_options = {
        'Sex_BMI': {
            'Y': bmi + sex
        },
        'Anth': {
            'none': [],
            'CA': m_manual,
            'DA_1': m_common,
            'DA_2': m_all
        },
        'ER': a_b_common,
        'Vols': volumes,
        'SA': SA_all
    }
    selector_options = {
        **{
            'Corr_' + str(n): lambda df, feature_cnames, target_cname:
            select_n_best_metric(df, feature_cnames, target_cname, n=n)
            for n in [2]},
        'mRMR': lambda df, feature_cnames, target_cname:
            mRMR(df, feature_cnames, target_cname, return_best=1)[0, 1].tolist()
    }
    # results = select_and_nonsequ_MFA(df, feature_options, all_targets, model, selector_options, selection_groups = ['Vols','SA'], ignored_groups = ['Sex_BMI','Anth'], sexes=[0,1,'all'])
    # print(results)
    # results.to_excel('results_4_3_20_mlp.xlsx', index=False)
    # # Condense
    # # size_lst = [1, 4, 0 + 2, 0 + 2 + #selector_options, 0 + 2 + #selector_options] = [1,4,2,4,4]
    # # choice_offset_lst = [1, 1*4, 1*4*2, 1*4*2*4, 1*4*2*4*4]
    # # 128 = Product(size_lst)
    # df = pd.read_excel('results_4_3_20_mlp.xlsx')
    # del df['DataSet']
    # df1 = df[df.index % 128 == 0]
    # df1.insert(0, 'Feature Set', 1)
    # df2 = df[df.index % 128 == 32]
    # df2.insert(0, 'Feature Set', 2)
    # df3 = df[df.index % 128 == 64]
    # df3.insert(0, 'Feature Set', 3)
    # df4 = df[df.index % 128 == 96]
    # df4.insert(0, 'Feature Set', 4)
    # df5 = df[df.index % 128 == 112]
    # df5.insert(0, 'Feature Set', 5)
    #
    # df6_list = [df[df.index % 128 == n].reset_index(drop=True) for n in [116, 120]]
    # df6 = pd.concat(df6_list).groupby(level=0).max(level='avg_test_r2')
    # df6.insert(0, 'Feature Set', 6)
    #
    # df7_list = [df[df.index % 128 == n].reset_index(drop=True) for n in [119, 118, 121, 122]]
    # df7 = pd.concat(df7_list).groupby(level=0).max(level='avg_test_r2')
    # df7.insert(0, 'Feature Set', 7)
    #
    # df8 = df[df.index % 128 == 127]
    # df8.insert(0, 'Feature Set', 8)
    # # besties = df[(df.Anth == 'DA_2') & (df.ER == 'all')].groupby(['Target','DEM_SEX'])['avg_test_r2'].agg({'avg_test_r2' : ['max']})
    # # print(besties)
    # results = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])
    # print(results)
    # results.to_excel('results_4_3_20_compact_mlp.xlsx', index=False)


    # g = sns.pairplot(df[volumes+['TOTAL_FAT']])

    # df = pd.read_csv('C:\\Users\\Clint\\PycharmProjects\\body-shape-ml\\python\\PCA_App\\pca_ds.csv')
    # targets = ['TOTAL_FAT', 'TOTAL_LEAN']
    # features = ['PC' + str(n) for n in range(1, 30 + 1)]
    # # df = df[['SubjectID'] + targets + features + ['SEX']].dropna()

    # for v6 encoding appears that 0 is F, 1 is M
    # results = select_and_run_targs_MF(df, all_features, all_targets, model, upto=10, graph_r2=True, sexes=[0, 1])
    # print(results)
    # results['n_PCS'] = results.index.map(lambda x: x + 1)
    # results = results.reset_index(drop=True)
    # del results['estimator']
    # del results['DataSet']
    # all_cols = [c for c in results.columns if c.startswith('all_')]
    # results2 = pd.DataFrame()
    # for n in range(len(results)):
    #     df1 = results.iloc[[n]]
    #     alls = df1[all_cols]
    #     n_splits = 5
    #     df2 = pd.DataFrame()
    #     for i in range(n_splits):
    #         dict = {}
    #         for col in all_cols:
    #             dict = {**dict, **{col[4:]: alls[col].iloc[0][i]}}
    #         df2 = append_dict(df2, dict)
    #     for df in [df1, df2]:
    #         df['m'] = 0
    #     r00 = pd.merge(df1, df2, on=['m'])
    #     results2 = pd.concat([results2, r00])
    # for col in ['m'] + all_cols:
    #     del results2[col]
    # results2.to_csv('fig_2.csv', index=False)
    # print(results2)

    # def write_model(inter, coefs, vars):
    #     inter = round(inter, 2)
    #     coefs = [round(x, 2) for x in coefs]
    #     terms = [inter] + [str(c) + ' ' + var for c, var in zip(coefs, vars)]
    #     return ' + '.join(map(str, terms))
    # print(results.apply(lambda x: write_model(x['intercept'], x['coef'], x['Selected'].split(', ')), axis=1))

    # cuts = [
    #     cut_by_IQR(df, q=2, target_cnames=['BC_DXA_FAT_TOT']),
    #     cut_by_IQR(df, q=1.5, target_cnames=['BC_DXA_LST_TOT'])
    # ]
    # idx = df.index
    # for c in cuts:
    #     idx = idx.intersection(c.index)
    # df_noout = df.iloc[idx]
    # df_noout.to_excel('../data/ShapeUp/AllStyku_noout_v14.xlsx', index=False)


if __name__ == "__main__":
    main()