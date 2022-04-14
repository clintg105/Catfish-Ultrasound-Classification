from typing import Mapping, Iterable, List, Tuple
from joblib import Parallel

import pandas as pd, datetime
__author__ = "Pujan Shrestha, Alex Mensen-Johnson, Clinten Graham, and Caleb Johnson"
__status__ = "Development"

'''
Type-Casting Functions
'''


def require_tuple(obj):
    if isinstance(obj, str):
        return (obj,)
    assert isinstance(obj, Iterable)
    return tuple(obj)


def require_dict(obj):
    if isinstance(obj, dict):
        return obj


def require_enumerable(item):
    if isinstance(item, (List, Tuple)):
        return item
    return [item]


'''
List and DataFrame Manipulation
'''


# Partition Python list with optional offset.
def partition(list, n, offset=0):
    return [list[i:i + n] for i in range(0, len(list), n - offset)]


def append_dict(df, dict):
    # append dictionary to dataframe
    dict_df = pd.DataFrame(data={key: {0: value} for key, value in dict.items()})
    return pd.concat([df, dict_df], sort=False)


def df_append(master_df, df, column_map):
    df = df.copy()
    for column_name, column_value in column_map.items():
        df[column_name] = column_value
    if master_df is not None:
        master_df = master_df.append(df)
        master_df.reset_index(drop=True, inplace=True)
    else:
        master_df = df
    return master_df


def df_reorder_columns(df, send_front=[], send_back=[]):
    middle = [col for col in df.columns if col not in send_front and col not in send_back]
    return df.reindex(columns=send_front + middle + send_back, copy=False)


def dataframe_with_structure(dataframe, ndarray):
    '''
    Takes an ndarray and converts it to a dataframe
    '''
    if isinstance(dataframe, pd.Series):  # checking for type series
        dataframe = dataframe.to_frame()  # converts series into a dataframe
    dataframe = dataframe.copy() # third possible irrelevant copy function?
    if (len(ndarray.shape) == 1):  # checks ndarray is vector (1-tensor)
        ndarray = ndarray.reshape(-1, 1) # reshapes array to be of option(-1) and size 1// -1 autofills based of the previous shape
    if len(dataframe.index) != ndarray.shape[0] or len(dataframe.columns) != ndarray.shape[1]:
        raise Exception("wrong shape")
    dataframe[dataframe.columns] = ndarray
    return dataframe


# Save data frame to csv file in reports
def save_df(df, save_name):
    df.to_csv(f"reports/{save_name}.csv", index=False)
    df.to_pickle(f"reports/{save_name}.pkl")


'''
Runtime Control
'''


# Resolve delayed statements with multicore processing
def resolve_delayed(delayed_iter, n_cores=-1):
    # Resolve delayed statements with multicore processing
    delayed_list = list(delayed_iter)
    return list(Parallel(n_jobs=n_cores)(delayed_list)) # Parallel - n_jobs = -1, use all cpu's


'''
Labeling Functions
'''


def timestamp():
    return int(datetime.datetime.now().strftime('%Y%m%d%H%M')[2:])


'''
Dot-Notation
'''


# compact way to use dot notation
class Map():
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def collect(self, exclude_keys=[]):
        all_values = []
        Map.__collect(self.__dict__, exclude_keys, all_values)
        return all_values

    @staticmethod
    def __collect(dict, exclude_keys, l):
        for key, val in dict.items():
            if key in exclude_keys:
                continue
            if isinstance(val, str):
                l.append(val)
            else:
                Map.__collect(val.__dict__, exclude_keys, l)