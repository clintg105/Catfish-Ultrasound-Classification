import pandas as pd, numpy as np, os, warnings, re
from datetime import date
from abc import abstractmethod
from os import path

from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
__author__ = "Pujan Shrestha, Alex Mensen-Johnson, Clinten Graham, and Caleb Johnson"
__status__ = "Legacy"


'''
Functions Used in Dataset Creation
'''


# Partition Python list with optional offset.
def partition(list, n, offset=0):
    return [list[i:i + n] for i in range(0, len(list), n - offset)]


# Create discrete class from numeric segmentation
def discrete_class(df, cname, segmentation, classnames=None, defaultclass='nan'):
    if classnames is None:
        classnames = list(range(len(segmentation) + 1))
    assert np.all(np.diff(segmentation) > 0), 'Segmentation must be strictly increasing.'
    assert len(classnames) == len(segmentation) + 1, \
        'Number of classes must be one more than the number of floats in segmentation.'

    internal_bounds = partition(segmentation, 2, 1)[:-1]
    conditions = [lambda x: x < segmentation[0]]
    conditions += [lambda x, l = lower, u = upper: (l <= x) & (x < u)
                   for lower, upper in internal_bounds]
    conditions += [lambda x: segmentation[-1] <= x]
    condlist = lambda x: [cnd(x) for cnd in conditions]

    return np.select(condlist(df[cname]), classnames, default=defaultclass)


def standardize_subject_ids(series):
    seen = []
    id_len = len("02ADL0153")  # standard length of ids

    def map_name(name):
        if name is not str:
            name = str(name)
        name = name.upper()
        id = name[0:id_len]  # takes name column and extracts ids for beginning of string
        seen.append(id)
        if len(name) > id_len:
            suffix = name[id_len:id_len + 2]  # next 2 characters after id
            if suffix == "_A":
                return id
            elif suffix == "_B":
                return id + "_2"
        return id if seen.count(id) == 1 else f"{id}_{seen.count(id)}"  # possibilities: id, id_2, id_n
    return series.apply(map_name)
def cut_subject_ids(series):
    id_len = len("02ADL0153")
    def map_name(name):
        name = name.upper()
        id = name[0:id_len]
        return id
    return series.apply(map_name)


'''
DataSet Creation and Modification Code
'''


class DataSet:
    def __init__(self):
        self.df = self.load_data()

    @abstractmethod
    # Processing Methods
    def load_data(self):
        pass

    def extract_data(self,
                     feature_cnames,
                     target_cname,
                     scaler_config={},
                     data_transformers=[],
                     blacklist_sids=[]):
        df = self.load_data().copy()
        # df = df.set_index('SubjectID', verify_integrity=True)
        # df = df[~df.index.isin(blacklist_sids)]

        feature_cnames = feature_cnames.copy()
        df = self.__transform(df, data_transformers, feature_cnames)
        df = self.__prune(df, feature_cnames + [target_cname])
        x = df[feature_cnames]
        y = df[[target_cname]]
        return ExtractedData(x, y, DataFrameScaler(scaler_config))

    def __transform(self, df, data_transformers, feature_cnames):
        for transformer in data_transformers:
            ret = transformer(df, feature_cnames)
            if isinstance(ret, pd.DataFrame):
                df = ret
        return df

    def __prune(self, df, cnames_to_keep):
        # abort if columns are missing
        missing_cnames = [cname for cname in cnames_to_keep if cname not in df.columns.values]
        if len(missing_cnames) > 0:
            print(f"Missing columns: {missing_cnames}")
            exit()

        # remove columns not in use
        df = df[cnames_to_keep]

        # remove rows with empty values
        df = df.replace([0, "", "nan"], None).dropna()
        return df

    # Import Methods
    # Returns a datasheet without duplicates. The duplicates are also aggregated for the different values.
    # This goes into StykuDataset
    def common_dataframes(self, include_classes=False, prune_hist=True, prune_fast=True, n_classes=2):
        # questionnaire_df = pd.read_csv(GITPATH + 'python/data/ShapeUp/common/Questionnaire.csv')
        # questionnaire_df['SubjectID'] = cut_subject_ids(questionnaire_df['SubjectID'])
        demographics_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx", sheet_name='Demographics')
        demographics_df['SubjectID'] = cut_subject_ids(demographics_df['SubjectID'].astype(str))
        # demographics_df['Race'] = demographics_df['Race'].map(lambda x: x.split(' ')[0] if isinstance(x, str) else x)

        # dexa_df = pd.read_excel(GITPATH + "python/data/ShapeUp/common/DXAnooutliers.xlsx", na_values=['#N/A'])
        dexa_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_181101.xlsx", sheet_name='DXA')
        dexa_df = dexa_df.dropna(axis=0, subset=['TOTAL_FAT', 'TOTAL_LEAN', 'TOTAL_PFAT'])
        dexa_df['TRUNK_BMC'] = dexa_df.apply(
            lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
        dexa_df['TOTAL_PLEAN'] = dexa_df.apply(lambda x: x['TOTAL_LEAN']/x['TOTAL_MASS'], axis=1)
        dexa_df['SubjectID'] = cut_subject_ids(dexa_df['SubjectID'])

        manual_df = pd.read_csv(GITPATH + "python/data/ShapeUp/common/Manual.csv")
        # manual_df = pd.read_excel(GITPATH + 'python/data/ShapeUp/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx', sheet_name='Manual'),
        manual_df['SubjectID'] = cut_subject_ids(manual_df['SubjectID'])

        # blood_df = pd.read_csv(GITPATH + "python/data/ShapeUp/common/Blood.csv")
        blood_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_181101.xlsx", sheet_name='Blood')
        blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])
        if include_classes:
            if prune_hist:
                # Remove people with family histories for HBA1C
                blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])
                blood_df = blood_df.merge(
                    pd.read_excel(GITPATH + 'python/data/ShapeUp/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx', sheet_name='History'),
                    on='SubjectID', how='outer')
                blood_df = blood_df.loc[blood_df['Fam_Diabetes'] != 'Yes']
                blood_df['SubjectID'] = standardize_subject_ids(blood_df['SubjectID'])
            if n_classes == 2:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40])  # 1 (>40) is healthy
            if n_classes == 3:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100, 125])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6, 6.4])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130, 160])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40, 60])  # 2 (>60) is healthy

        bia_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_181101.xlsx", sheet_name="BIA")
        # bia_df = bia_df.rename(columns={"_BFM_Body_Fat_Mass_": "TOTAL_FAT", "_LBM_Lean_Body_Mass_": "TOTAL_LEAN"})
        # bia_df['_AGE'] = bia_df['_AGE'].astype(float)
        bia_df = bia_df.groupby(['SubjectID'], as_index=False).aggregate('mean')

        #a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")
        #a_over_b_df['SubjectID'] = cut_subject_ids(a_over_b_df['SubjectID'])

        dfs = [#questionnaire_df,
               demographics_df, dexa_df, manual_df, blood_df, bia_df]
        combined_df = None
        for df in dfs:
            df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='SubjectID', how='outer')

        combined_df['age'] = combined_df['BIRTHDATE'].astype(str).map(
            lambda row: date.today().year - int(row[2:4]) - 1900 if row != 'NaT' else "")
        # combined_df['age'] = combined_df.apply(
        #     lambda row: date.today().year - row['qff_dob_year'] if row['age'] == "" else row['age'], axis=1)
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')

        if include_classes and prune_fast:
            # Remove subjects that did not fast
            combined_df = combined_df.loc[(combined_df['TRIG'] <= 180) | (combined_df['BMI1'] >= 45)]  # Remove subjects w/ TRIG > 180 & BMI < 45

        return combined_df

    # Returns a merged common datasheet where dexa has duplicate subject ids and the others are duplicated per subject id.
    # This goes into StykyDataset_2
    def common_dataframes_2(self, include_classes=False, prune_hist=True, prune_fast=True, n_classes=2):
        questionnaire_df = pd.read_csv(GITPATH + 'python/data/ShapeUp/common/Questionnaire.csv')
        dexa_df = pd.read_csv(GITPATH + "python/data/ShapeUp/common/DXA.csv", na_values=["#N/A"])
        dexa_df['TRUNK_BMC'] = dexa_df.apply(
            lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
        dexa_df['TOTAL_PLEAN'] = dexa_df.apply(lambda x: x['TOTAL_LEAN']/x['TOTAL_MASS'], axis=1)
        manual_df = pd.read_csv(GITPATH + "python/data/ShapeUp/common/Manual.csv")
        blood_df = pd.read_csv(GITPATH + "python/data/ShapeUp/common/Blood.csv")
        if include_classes:
            if prune_hist:
                # Remove people with family histories for HBA1C
                blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])
                blood_df = blood_df.merge(
                    pd.read_excel('data/ShapeUp/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx', sheet_name='History'),
                    on='SubjectID', how='outer')
                blood_df = blood_df.loc[blood_df['Fam_Diabetes'] != 'Yes']
                blood_df['SubjectID'] = standardize_subject_ids(blood_df['SubjectID'])
            if n_classes == 2:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40])  # 1 (>40) is healthy
            if n_classes == 3:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100, 125])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6, 6.4])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130, 160])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40, 60])  # 2 (>60) is healthy
        # NEW: Creating numeric health classes
        #blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6, 6.4])
        #blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100, 125])
        #a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")

        combo = blood_df.merge(manual_df, how='outer', on='SubjectID')
        #combo = combo.merge(a_over_b_df, how='outer', on='SubjectID')
        combo = combo.merge(questionnaire_df, how='outer', on='SubjectID')
        combo = dexa_df.merge(combo, how='outer', on='SubjectID', copy=True)
        if include_classes and prune_fast:
            # Remove subjects that did not fast
            combo = combo.loc[(combo['TRIG'] <= 180) | (combo['BMI1'] >= 45)]  # Remove subjects w/ TRIG > 180 & BMI < 45
        return combo

    # def search_update(self, prefix):
    #     Path = PathMan()
    #     strange_path = Path.getter() + "python\data"
    #     #print(strange_path)
    #     dir = DirGrab(strange_path)
    #     dir.grabFromPrefix(prefix)
    #     files = dir.getter()
    #     Search = Searcher()
    #     file_list = Search.GreatestValue(files)
    #
    #     #print(file_list[0])
    #     dataset_location = file_list[0]
    #     dataset_df = pd.read_excel(dataset_location, na_values=["[]", 0])
    #     return dataset_df





'''
Data Extraction and Column Scaler Code
'''


class ExtractedData:
    def __init__(self, x, y, scaler):
        self.x = x  # x column
        self.y = y  # y column
        self.x_scaled = scaler.fit_transform(x)
        self.y_scaled = scaler.fit_transform(y)
        self.scaler = scaler # scalar


class DataFrameScaler:
    def __init__(self, scaler_dict, save_scalar_params=False):
        '''
        This function is passed scalar_dict of form {cname: scalar function} where cname
        is either the name of a column in some dataframe or 'default'. Then the transform
        function is applied to the dataframe df; each column is scaled appropriately.
        '''
        self.default_scaler = None
        self.column_scalers = {} # dictionary
        if "default" in scaler_dict and scaler_dict["default"] is not None:
            self.default_scaler = scaler_dict["default"]
        for column_tuple, scaler_class in scaler_dict.items():
            if column_tuple == "default":
                continue
            scaler = scaler_class() if scaler_class is not None else None #scalar assingment eithe none or key value

            # verify column_tuple is iterable
            if not isinstance(column_tuple, tuple):
                column_tuple = (column_tuple,)

            for column in column_tuple:
                self.column_scalers[column] = scaler # dictionary assingment

    def __get_column_scaler(self, column):  # sets self.column_scalers[column] to default or other entry in scaler_dict
        if column not in self.column_scalers and self.default_scaler is not None:
            self.column_scalers[column] = (self.default_scaler)()
        if column in self.column_scalers and self.column_scalers[column] is not None:
            yield self.column_scalers[column]

    def transform(self, df):  # Apply scalars to columns
        df = df.copy()
        for column in df.columns:
            for scaler in self.__get_column_scaler(column):  # Returns list of scalars to transform column
                df[column] = scaler.transform(df[column].to_frame())
        return df

    def fit(self, df):  # fit scalars
        for column in df.columns:  # enumerate of str cnames
            for scaler in self.__get_column_scaler(column):  # Returns list of scalars to fit to column
                frame = df[column].to_frame()  # convert column to df
                if hasattr(scaler, "partial_fit"):
                    scaler.partial_fit(frame)
                else:
                    scaler.fit(frame)  # fit scalar to df, e.g. compute the mean and std to be used for later scaling.

    def fit_transform(self, df):
        self.fit(df)
        df = self.transform(df)
        return df

    def inverse_transform(self, df):  # Scale the columns back to their original representation
        for column in df.columns:
            for scaler in self.__get_column_scaler(column):  # Returns list of scalars to transform column
                df[column] = scaler.inverse_transform(df[column].to_frame())
        return df


'''
Other DataSet Functions
'''


# Create a DataSet from a pandas dataframe
def to_DataSet(df, combine_common=False, subject_cname='SubjectID'):
    if not (combine_common):
        class AutoDataset(DataSet):
            def __init__(self):
                self.df = df
            def load_data(self):
                return self.df
        return AutoDataset()
    class AutoDataset(DataSet):
        def load_data(self):
            try:
                # find subject IDs and combine with DXA, Blood, and Questionaire data
                df['SubjectID'] = cut_subject_ids(df[subject_cname])  # remove scan suffixes
                if subject_cname != 'SubjectID':
                    del df[subject_cname]  # The next step expects numeric columns

                # TODO: Combine label columns differently than numeric columns
                clean_df = df.replace([0, "[]", "", "nan"], None).dropna()
                clean_df = clean_df.drop_duplicates('SubjectID')
                # clean_df = clean_df.groupby(clean_df['SubjectID'], as_index=False).aggregate('mean')  # take average of duplicates

                combined_df = super().common_dataframes()
                combined_df = combined_df.merge(clean_df, on='SubjectID', how='outer')
                return combined_df
            except KeyError:
                # if there are no subject identifiers, do not combine with DXA
                warnings.warn(f'Could not find subject identifier column \"{subject_cname}\", loading without DXA.', stacklevel=4)
                return df
    return AutoDataset()


def hold_dataset(dataset, rewrite=False):
    name = type(dataset).__name__
    if not os.path.exists('./extracted datasets'):
        os.makedirs('./extracted datasets')
    if not os.path.isfile(f"./extracted datasets/{name}") or rewrite:
        dataset.load_data().to_csv(f"./extracted datasets/{name}.csv")
# hold_dataset(StykuDataSet(include_classes=True), rewrite=True)
# df = pd.read_csv('./extracted datasets/StykuDataSet.csv')
# df = df.loc[df['SEX'] == 'M']


def require_extracted(df_path: str, cnames=None) -> pd.DataFrame:
    ''' :returns extracted df if already present, otherwise extracts and saves '''
    split_path = re.split(r'\\|/|\.', df_path)
    file_name, file_ext = split_path[-2], split_path[-1]

    ext_df_path = df_path[:-(len(file_name)+len(file_ext)+1)] + 'ext-' + file_name + '.csv'
    if path.exists(ext_df_path):
        return pd.read_csv(ext_df_path)
    else:
        read_func = {'xlsx': pd.read_excel, 'csv': pd.read_csv}[file_ext]
        df = read_func(df_path)

        scalar_config = {"DEM_SEX": LabelBinarizer, "DEM_AGE": MinMaxScaler, "default": StandardScaler}

        if cnames is None:
            all_feats = df.columns.values.tolist()
            all_feats.remove('SubjectID')
        else:
            all_feats = cnames
        df['old_idx'] = df.index.values  # serves also as fake numeric 'target'
        df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
        df = to_DataSet(df).extract_data(all_feats, 'old_idx', scaler_config=scalar_config, data_transformers=[]).x_scaled
        df['SubjectID'] = df.index
        df.to_csv(ext_df_path)
        return df