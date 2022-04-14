from functools import reduce
from typing import Mapping
from joblib import delayed
from common_functions import require_tuple, resolve_delayed

import pandas as pd, numpy as np
__author__ = "Clinten Graham"  # 7/19
__maintainer__ = "Clinten Graham"  # 2/20
__email__ = "clintengraham@gmail.com"
__status__ = "Prototype"

'''
DataGrid Creation
'''

def dict_to_datagrid(option_dict, cname=None):
    '''
    Example: option_dict={'common': 1, 'all': [2, 3], 'none': 0}, cnames='measurements'

      measurements __data__
    0       common        1
    1          all   [2, 3]
    2         none        0
    '''
    return option_datagrid({cname: option_dict})

def option_datagrid(option_dict):
    '''
    Create DataGrid from option dictionary.

    :param option_dict: A singleton dictionary of the form {'OptName': {'Opt1': 'Val1', ...}}
    :return: DataGrid with columns {'OptName': ['Opt1', ...], '__data__': ['Val1', ...]}
    '''
    cname = list(option_dict)[0]
    option_dict = option_dict[cname]
    if isinstance(option_dict, pd.DataFrame):
        return option_dict
    if cname is None or "__cols" in option_dict:
        cname = require_tuple(option_dict["__cols"])
    else:
        cname = require_tuple(cname)

    # Make empty columns
    dict = {}
    for col in cname:
        dict[col] = []
    data = []
    if not isinstance(option_dict, Mapping): # mapping = dict
        assert len(cname) == 1
        option_dict = {str(item): item for item in option_dict}
    for key, value in option_dict.items():
        if key != "__cols":
            key = require_tuple(key)
            for idx in range(len(key)):
                dict[cname[idx]].append(key[idx])
            data.append(value)
    return pd.DataFrame({**dict, "__data__": data})

def option_datagrid_list(cname_to_option_dicts):
    '''
    Generalizes option_datagrid for multi-entry option_dicts.

    :param cname_to_option_dicts: Nested dictionary mapping option names to option dictionaries.
    :return: List of option DataGrids.
    '''
    dfs = []
    for cname, option_dict in cname_to_option_dicts.items():
        assert isinstance(option_dict, Mapping) # Make sure its a dict
        dfs.append(option_datagrid({cname: option_dict}))
    return dfs


def singleton_datagrid(obj, name):
    # Returns single-column datagrids for arbitrary python objects and lists.
    if isinstance(obj, list):
        if isinstance(obj[0], str):
            return dict_to_datagrid({x: x for x in obj}, name)
        else:
            return dict_to_datagrid({type(x).__name__: x for x in obj}, name)
    else:
        if isinstance(obj, str):
            return dict_to_datagrid({obj: obj}, name)
        else:
            return dict_to_datagrid({type(obj).__name__: obj}, name)


# DictGrids are standard DataGrids with dictionary entries in __data__
def compose_dictgrid(param_option_dict):
    '''
    Enumerate 'decision tree' branches as rows of a pd DataFrame from a param_option_dict.

    :param param_option_dict: Dictionary with keys as categories and values as options for those categories
    :return: Pandas DataFrame with columns from keys in param_option_dict and a __data__ column with the row data as a dict
    '''
    dfs = []
    for key, option_dict in param_option_dict.items():
        if not isinstance(option_dict, pd.DataFrame):
            # convert into dataframe
            option_dict = option_datagrid({key: option_dict})
        # check that it's a dataframe. assert aborts and gives an error if it's false
        assert isinstance(option_dict, pd.DataFrame)
        # writes dfs -  a list with vales from __data__ and keys from dict in {} pandas
        dfs.append(option_dict.assign(__data__=option_dict["__data__"].apply(lambda val: {key: val})))
    if len(dfs) == 0:
        return pd.DataFrame()
    return list_product(lambda x, y: {**x, **y}, dfs)

def unpack_dictgrid(dictgrid):
    # WARNING: All entries in dict grid are expected to have SAME KEYS.
    keys = dictgrid['__data__'][0].keys()
    for key in keys:
        dictgrid[key] = dictgrid['__data__'].map(lambda dict: dict[key])
    return dictgrid.drop('__data__', axis=1)


'''
DataGrid Functions
The code below is used to map arbitrary functions over DataGrids.
'''


def eval_product(fn, *dgs, n_cores=1):
    '''
    Takes 'Cartesian Product' of DataGrids and maps fn over their __data__ entries.

    :param fn: Function with sequential inputs to map over __data__ columns
    :param dgs: Ordered sequence of DataGrids
    :param n_cores: Number of processor cores to use in evaluation; -1 uses all.
    :return: DataGrid 'product' with product.__data__ = fn(dg1.__data__ x dg2.__data__ x ...)
    '''
    df_list = []
    for df in dgs:
        df_list.append(df.assign(__merge__=1))
    product = reduce(lambda df1, df2: df1.merge(df2, on="__merge__"), df_list)
    nondata_cnames = [cname for cname in product.columns if 'data' not in cname]
    product_data = product.drop(nondata_cnames, axis=1)
    product = product[nondata_cnames]
    product['__data__'] = product_data.apply(require_tuple, axis=1) # apply require_tuple over rows
    if n_cores != 1:
        product['__data__'] = resolve_delayed([delayed(fn)(*tupl) for tupl in product['__data__']], n_cores=n_cores)
    else:
        product['__data__'] = [fn(*tupl) for tupl in product['__data__']]
    return product.drop('__merge__', axis=1)


def binary_product(fn, df1, df2):
    # Returns product = fn(df1 x df2) (i.e. product.__data__ is image of fn(df1.__data__ x df2.__data__))
    df1 = df1.assign(__merge__=1)
    df2 = df2.assign(__merge__=1)
    product = df1.merge(df2, on="__merge__")
    product["__data__"] = [fn(x, y) for x, y in zip(product["__data___x"].values, product["__data___y"].values)]
    product.drop(["__merge__", "__data___x", "__data___y"], inplace=True, axis=1)
    return product


def list_product(fn, dgs: list):
    # For binary associative fn (fn forms monoid over dgs), returns left-folded fn( ... fn(fn(df1 x df2) x df3) x ...)
    return reduce(lambda x1, x2: binary_product(fn, x1, x2), dgs)  # reduce: f(x1, x2, x3) = f(f(x1, x2), x3)


def eval_inplace(fn, *dgs, inherit_cols=0, n_cores=1):
    '''
    Maps fn over DataGrids __data__ entries inplace.

    :param fn: Function with sequential inputs to map over __data__ columns.
    :param dgs: Ordered sequence of DataGrids with the same number of ordered rows.
    :param inherit_cols: Index of dg in dgs to inherit non-__data__ columns from.
    :param n_cores: Number of processor cores to use in evaluation; -1 uses all.
    :return: DataGrid 'product' with product.__data__ = fn(dg1.__data__, dg2.__data__, ...)
    '''
    data_list = np.array([dg.__data__.values for dg in dgs]).T
    data_list = [require_tuple(s) for s in data_list]
    product = dgs[inherit_cols]
    if n_cores != 1:
        product["__data__"] = resolve_delayed([delayed(fn)(*tupl) for tupl in data_list], n_cores=n_cores)
    else:
        product["__data__"] = [fn(*tupl) for tupl in data_list]
    return product


# dg1 = pd.DataFrame({'col1': ['A','B'], '__data__': [[1],[]]})
# dg2 = pd.DataFrame({'col2': ['A','B'], '__data__': [[2],[]]})
# dg3 = pd.DataFrame({'col3': ['A','B'], '__data__': [[3],[]]})
# print(eval_product(lambda x, y, z: x + y + z, dg1, dg2, dg3))
# print(eval_inplace(lambda x, y, z: x + y + z, dg1, dg2, dg3, inherit_cols=0))

'''
start = timeit.default_timer()
dg = eval_dgrid(lambda target, trnsfrm, cnames: dataset.extract_data(
                cnames, target, scaler_config=scalar_config_dict, data_transformers=trnsfrm),
           target_dg, trnsfrm_dg, cname_dg)
stop = timeit.default_timer()
print('Time: ', stop - start)

start = timeit.default_timer()
dg = eval_dgrid(lambda target, trnsfrm, cnames: dataset.extract_data(
                cnames, target, scaler_config=scalar_config_dict, data_transformers=trnsfrm),
           target_dg, trnsfrm_dg, cname_dg, multicore=True)
stop = timeit.default_timer()
print('Multicore Time: ', stop - start)

# Time: 41.5880947 
# Multicore Time: 26.3387945
'''

# df1 = dict_to_datagrid({'common': [1], 'all': [1, 2, 3], 'none': []},
#                        cname='measurements')
# print('df1: \n {}'.format(df1))
#
# df2 = option_datagrid({'Volumes': {'common': [4], 'all': [4, 5], 'none': []}})
# print('df2: \n {}'.format(df2))
#
# df3 = compose_dictgrid({'Volumes':
#                             {'common': [4], 'all': [4, 5], 'none': []},
#                         'Measurements':
#                             {'common': [1], 'all': [1, 2, 3], 'none': []}
#                         })
# print('df3: \n {}'.format(df3))
#
# df0 = nary_product(lambda x1, x2: x1 + x2,
#                    df1, df2)
# print('df0: \n {}'.format(df0))
#
# start = timeit.default_timer()
# df0 = nary_product(lambda x: {'Values': x}, df1)
# df0 = unpack_dictgrid(df0)
# print('df0: \n {}'.format(df0))
# stop = timeit.default_timer()
# print('Time: ', stop - start)
#
# start = timeit.default_timer()
# df0 = df1.rename(columns={'__data__': 'Values'})
# print('df0: \n {}'.format(df0))
# stop = timeit.default_timer()
# print('Time: ', stop - start)
#
# df0 = nary_product(lambda x1, x2: {'Out 1': x1 + x2, 'Out 2': 2*x1 +  x2},
#                    df1, df2)
# print('df0: \n {}'.format(df0))
# df0 = unpack_dictgrid(df0)
# print('df0: \n {}'.format(df0))