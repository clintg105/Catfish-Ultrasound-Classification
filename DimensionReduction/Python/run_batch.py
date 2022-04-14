import os, csv, uuid, numpy as np, pandas as pd

import dataframe.datagrid as dg
from dataframe.datasets import ExtractedData, to_DataSet
from common_functions import append_dict, partition, df_reorder_columns, timestamp, require_tuple

import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score, SCORERS, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RepeatedStratifiedKFold, ShuffleSplit
from joblib import dump, load  # Similar to pickle, optimized for objects with large internal numpy arrays
__author__ = "Clinten Graham"
__credits__ = ["Clinten Graham", "Pujan Shrestha", "Sima Sobhiyeh",
               "Peter Wolenski", "Caleb Johnson"]
__maintainer__ = "Clinten Graham"
__email__ = "clintengraham@gmail.com"
__status__ = "Development"


def run_batch(dataset,
              data_config,
              models,
              model_param_grid={},
              dataset_name=None,
              ext_dataset=None,
              ext_dataset_name=None,
              eval_type='regressor',
              cv_params={},
              show_best_runs=None,
              save_prompt=False,
              n_cores=-1
              ):
    '''
    Multi-threaded hyperparameter searches using DataGrids.
    TODO: Test tf functionality (robust or bust!)
    TODO: Add progress indicators and a verbosity option to either run_batch or eval_product.

    :param dataset: List of DataSet type objects with identical feature names.
    :param data_config: Dictionary containing information on how the datasets should be extracted and scaled.
        :key target_cnames: List of column names to be used as targets.
        :key feature_options: {group_name: {option: [features]}}
            Dictionary of feature groups, each containing 'option: [features]' key-value pairs.
        :key transform_options: {feature: {category: [transformers]}}
            Dictionary of categorical features to split the run over, for each possible category there is a list of
            data transformers. Ideal for sex and ethnicity.
        :key scalar_config: Dictionary containing 'feature: scalar' key-value pairs for sklearn formatted scalars.
    :param models:
    :param model_param_grid:
    :param ext_dataset:
    :param eval_type: String indicating if run is 'regressor' (default) or 'classifier' (may use 'r' or 'c' resp.)
    :param cv_params: Dictionary of parameters to be passed to cv_method, which is either classifierCV or regressorCV.
        (see possible arguments in 'Cross Validation and Scoring Functions' section below)
    :param show_best_runs: Int indicating the best n runs to print for each target and sex option
    :param save_prompt: Whether to prompt the use if best runs shown (from :show_best_runs) should be saved
    :param n_cores: Number of processor cores to use in evaluation over DataGrids; -1 (default) uses all
    :return: DataFrame containing run information, scoring information and estimators (trained models)
    '''
    # options for dataframe input
    if isinstance(dataset, pd.DataFrame):
        dataset = to_DataSet(dataset)
    # setup cross-validation and scoring method, different for regression and classification
    if eval_type[0] == 'c':
        cv_method = classifierCV
        score_method = classifierScore
    else:
        cv_method = regressorCV
        score_method = regressorScore

    # create DataGrids
    dataset_dg = dg.singleton_datagrid(dataset, 'DataSet')
    target_dg = dg.singleton_datagrid(data_config['target_cnames'], 'Target')
    if data_config['transform_options'] != {}: # UGLY if statement wtf
        trnsfrm_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config['transform_options']))
    else:
        trnsfrm_dg = [0]
    cname_dg = dg.list_product(lambda x, y: x + y, dg.option_datagrid_list(data_config['feature_options']))
    cname_dg = cname_dg[(cname_dg['__data__'].map(len) != 0)]  # Remove rows with no input columns

    # extract data to ExtractedData type object
    print('[STATUS] Extracting Data')
    if data_config['transform_options'] != {}:
        data_dg = dg.eval_product((lambda dataset, target, trnsfrm, cnames:
            dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']), data_transformers=trnsfrm)),
                                  dataset_dg, target_dg, trnsfrm_dg, cname_dg, n_cores=n_cores)
    else:
        data_dg = dg.eval_product((lambda dataset, target, cnames:
            dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']))),
                                  dataset_dg, target_dg, cname_dg, n_cores=n_cores)

    # train, test and score models. There are three possibilities for input.
    print('[STATUS] Training Estimators')
    if model_param_grid != {}:
        # perform hyper-parameter grid search
        assert callable(models), 'regressor_param_grid cannot be used on multi-regressor runs.'
        hyperparam_dg = dg.compose_dictgrid(model_param_grid)
        regressor_dg = dg.eval_product(lambda params: models(**params), hyperparam_dg)
        results_dg = dg.eval_product((lambda data, reg: cv_method(data, reg, **cv_params)),
                                     data_dg, regressor_dg, n_cores=n_cores)
    elif isinstance(models, list):
        # train each regressor on the extracted data
        regressor_dg = dg.singleton_datagrid(models, 'Model')
        results_dg = dg.eval_product((lambda data, reg: cv_method(data, reg, **cv_params)),
                                     data_dg, regressor_dg, n_cores=n_cores)
    else:
        # train single regressor on extracted data
        results_dg = dg.eval_product((lambda data: cv_method(data, models, **cv_params)),
                                     data_dg, n_cores=True)
    # unpack final results
    results_df = dg.unpack_dictgrid(results_dg)

    if ext_dataset is not None:
        print('[STATUS] Extracting External Data')
        ext_dataset_dg = dg.singleton_datagrid(ext_dataset, 'ExtDataSet')
        ext_data_dg = dg.eval_product((lambda ext_dataset, target, trnsfrm, cnames:
            ext_dataset.extract_data(cnames, target, scaler_config=(data_config['scalar_config']), data_transformers=trnsfrm)),
                                      ext_dataset_dg, target_dg, trnsfrm_dg, cname_dg, n_cores=n_cores)

        # ensure that ext_data_dg has the same rows as results_df
        if model_param_grid != {}:
            ext_data_dg = dg.binary_product(lambda data, reg: data, ext_data_dg, hyperparam_dg)
        elif isinstance(models, list):
            ext_data_dg = dg.binary_product(lambda data, reg: data, ext_data_dg, regressor_dg)

        print('[STATUS] Training Estimators on External Data')
        results_df['__data__'] = results_df['estimator']
        results_df = dg.eval_inplace((lambda reg, data:
                                      {'ext_dataset': type(ext_dataset).__name__,
                                       **score_method(data, reg, prefix='ext')}
                                      ), results_df, ext_data_dg, n_cores=n_cores)

        # unpack final results
        results_df = dg.unpack_dictgrid(results_df)
        results_df = df_reorder_columns(send_back=['estimator'], df=results_df)

    if show_best_runs is not None:
        if isinstance(show_best_runs, int):
            metric_col = {'r': 'avg_test_r2', 'c': 'avg_test_accuracy'}[eval_type[0]]
        else:  # (cname, num) format
            metric_col = show_best_runs[1]
            show_best_runs = show_best_runs[0]
        # results_df['FPR0'] = results_df['FPR'].map(lambda x: 1 - x if x != 0 else x)  # used to sort by FPR for classification runs
        block_size = int(len(results_dg) / (len(dataset_dg) * len(target_dg) * len(trnsfrm_dg)))
        run_blocks = partition(results_df, block_size)


        if show_best_runs > 0:
            run_blocks = [df.nlargest(show_best_runs, columns=[metric_col]) for df in run_blocks]
        else:
            run_blocks = [df.nsmallest(-show_best_runs, columns=[metric_col]) for df in run_blocks]

        best_runs = pd.concat(run_blocks)
        print('[STATUS] Best Runs:\n', best_runs)
        save = str(input('[EXPORT] Save evaluators to disk (y/[n])? ')) if save_prompt else 'n'
        if save_prompt and save == 'y':
            # best_runs['Data'] = best_runs.index.map(lambda x: data_dg['__data__'][x % len(data_dg)])
            best_runs['Features'] = best_runs.index.map(lambda x: x % (len(cname_dg) * len(dataset_dg) * len(trnsfrm_dg)))
            uuid_dict = {feature_code: uuid.uuid1().hex for feature_code in best_runs['Features'].unique()}
            best_runs['UUID'] = best_runs['Features'].map(uuid_dict)
            best_runs['Features'] = best_runs['Features'].map(lambda x: cname_dg['__data__'][x % len(cname_dg)])
            data_dict = {type(DataSet).__name__: DataSet for DataSet in dataset}
            best_runs['DataSet'] = best_runs['DataSet'].map(data_dict)
            # print(f'[STATUS] Training and saving {type(model).__name__} on {target_cname}')
            print(best_runs)

    return results_df


def threshold_scan(thresholds,
                    datasets,
                    data_config,
                    classifiers,
                    model_param_grid={},
                    cv_params={},
                    ext_dataset=None,
                    show_best_runs=0,
                    n_cores=-1
                    ):
    '''
        Given a list of threshold values and a classifier with predict_proba, run a run_batch grid search and score the
    classifier at each threshold. All arguments except :thresholds are passed to run_batch.
    '''
    if isinstance(thresholds, int):
        thresholds = np.linspace(0, 1, thresholds)
    results = pd.DataFrame()
    for threshold in thresholds:
        print('[STATUS] Current threshold: ', threshold)
        batch_results = run_batch(
                    datasets,
                    data_config,
                    classifiers,
                    model_param_grid=model_param_grid,
                    ext_dataset=ext_dataset,
                    eval_type='classifier',
                    cv_params={**cv_params, **{'threshold': threshold}},
                    show_best_runs=show_best_runs,
                    n_cores=n_cores
        )
        batch_results.insert(0, 'threshold', threshold)
        results = pd.concat([results, batch_results])
    return results


def ext_train_save(model,
                   dataset,
                   feature_cnames,
                   target_cname,
                   scaler_config={},
                   eval_type='classifier',
                   cv_params={},
                   model_name=None,
                   timestamp=timestamp(),
                   save_location='models'
                   ):
    '''
    Extract data, train a model on the data, then save the model to ./models/

    :param model: an sklearn estimator
    :param dataset: a DataSet object from DataSets.py
    :param feature_cnames: list of column names to be used as features
    :param target_cname: column name of target
    :param eval_type: 'classifier' or 'regressor'
    :param model_name: name to save the model as (autogenerated by default)
    :param save_location: location to save the model (relative where this function is called from)
    '''
    cv_params = {**dict(total_train=True), **cv_params}  # ensure always total_train=True
    if eval_type == 'classifier':
        cv_method = classifierCV
    else:
        cv_method = regressorCV
    if model_name is None:
        model_name = type(model).__name__
    save_name = f"{target_cname}_{model_name}_{timestamp}"  # location and save name of extracted model
    Data = dataset if isinstance(dataset, ExtractedData) else dataset.extract_data(feature_cnames, target_cname, scaler_config)
    # train and score the model
    cv_score = cv_method(Data, model, **cv_params)
    # export the model
    dump(cv_score['estimator'], f"{save_location}/{save_name}_mdl.joblib")
    del cv_score['estimator']
    # export scalars
    if scaler_config != {}:
        if os.path.exists(f'{save_location}/{timestamp}_scalers.joblib'):
            scalers = load(f'{save_location}/{timestamp}_scalers.joblib')
            scalers.fit(Data.y)
            dump(scalers, f"{save_location}/{timestamp}_scalers.joblib")
        else:
            dump(Data.scaler, f"{save_location}/{timestamp}_scalers.joblib")
    # write model information to the models csv file
    if os.path.exists(f'{save_location}/models.csv'):
        f = open(f'{save_location}/models.csv', 'a+', newline='')
    else:
        f = open(f'{save_location}/models.csv', 'w+', newline='')
        f.write('Target, Model, Type, Name, Features, Scores \n')
    writer = csv.DictWriter(f, fieldnames=['Target', 'Model', 'Type', 'Name', 'Features', 'Scores'])
    writer.writerow({
        'Target': target_cname,
        'Model': model_name,
        'Type': eval_type,
        'Name': save_name,
        'Features': feature_cnames,
        'Scores': dict(cv_score)
    })
    f.close()


'''
Cross Validation and Scoring Functions
    Any function ended in 'Score' expects to score a pre-trained model for the :reg parameter.
'''
# scorer_dict = {**{
#     'RMSE':
# }, **SCORERS}

# Train a regressor and score it using cross-validation
def regressorCV(data, reg, n_splits=5, scorers=['r2'], total_train=False, return_params=False):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    cv_scores = cross_validate(reg, X, y, scoring=scorers, cv=n_splits,
                               return_train_score=True, return_estimator=True)
    estimator = cv_scores['estimator'][np.argmax(cv_scores['test_' + scorers[0]])]  # select by best test score
    del cv_scores['estimator']
    del cv_scores['fit_time']
    del cv_scores['score_time']
    cv_scores_avg = {'avg_' + key: sum(value) / n_splits for key, value in cv_scores.items()}
    cv_scores_std = {'std_' + key: value.std() for key, value in cv_scores.items()}
    cv_scores_all = {**cv_scores_avg, **cv_scores_std, **{'all_' + k: v for k,v in cv_scores.items()}}

    if total_train:
        reg0 = reg.fit(X, y)
        cv_scores_all = {**cv_scores_all, **(regressorScore(data, reg0, prefix='total_train_'))}
        del cv_scores_all['total_train_subjects']
        estimator = reg0
    cv_scores_all['estimator'] = estimator
    if return_params:
        cv_scores_all = {**cv_scores_all, **{'coef': estimator.coef_, 'intercept': estimator.intercept_}}
    return {**{'subjects':len(y)}, **cv_scores_all}


# Score a TRAINED regressor given standard scoring input
def regressorScore(data, reg, prefix=''):
    X = data.x_scaled.values
    y = data.y_scaled.values.ravel()
    columns = dict()
    columns['subjects'] = len(y)
    columns['r2'] = r2_score(y, reg.predict(X))
    if prefix != '':
        columns = {prefix + key: value for key, value in columns.items()}
    return columns


# Train a classifier and score it using cross-validation
def classifierCV(data,
                 clf,
                 scorers=['accuracy'],
                 cnd=None,  # condition index to be considered as 'positive' for 2x2 classification
                 threshold=None,
                 n_splits=5,
                 total_train=False,
                 external_validation=0,  # % of dataset to use for external validation
                 cv_method=StratifiedKFold  # options: https://scikit-learn.org/stable/modules/cross_validation.html
                 ):
    X = data.x_scaled.values
    if len(X) == 0:
        return {'subjects': 0}
    y = data.y.values.astype('int32').ravel()
    # try:
    #     y = data.y.values.astype('int32').ravel()
    # except ValueError:
    #     print(data.x.columns)
    #     print(data.y)
    #     exit()

    if external_validation != 0:
        X, X_ext, y, y_ext = train_test_split(X, y, test_size=external_validation, random_state=42)  # ext is 'test'

    cv_scores = pd.DataFrame()
    for train_idx, test_idx in cv_method(n_splits=n_splits).split(X, y):
        cv_cols = dict()
        X_train, X_test, y_train, y_test = (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
        clf0 = clf if threshold is None else threshold_clf(clf, cnd, threshold)
        clf0.fit(X_train, y_train)
        y_pred = clf0.predict(X_test)
        y_train_pred = clf0.predict(X_train)
        mtrx = confusion_matrix(y_test, y_pred)
        cv_cols['Confusion Matrix'] = mtrx
        for scorer in scorers:
            # Get scorer function (THERE HAS TO BE A BETTER WAY!)
            scr = str(SCORERS[scorer])
            scr = scr[scr.find("(")+1:scr.find(")")]
            scr = eval('skm.'+scr)
            cv_cols['train_' + scorer] = scr(y_train, y_train_pred)
            cv_cols['test_' + scorer] = scr(y_test, y_pred)
        # cv_cols['train_acc'] = accuracy_score(y_train, clf0.predict(X_train))
        # cv_cols['test_acc'] = accuracy_score(y_test, y_pred)
        if cnd is not None:
            cv_cols = {**cv_cols, **(cnd_score(mtrx, cnd, verbose=True))}
        cv_cols['estimator'] = clf0
        cv_scores = append_dict(cv_scores, cv_cols)

    #mean_cv_scores = cv_scores.mean(axis=0)
    mean_cv_scores = cv_scores[
        [cname for cname in cv_scores.columns if not(cname.endswith('Matrix') | cname.endswith('estimator'))]].mean()
    for cname in cv_scores.columns:
        if 'Matrix' in cname:
            mean_cv_scores[cname] = np.sum(cv_scores[cname].values) / n_splits
    # if cnd is not None:
    #     for cname in ('tp', 'fn', 'fp', 'tn'):
    #         mean_cv_scores[cname] = int(mean_cv_scores[cname] * n_splits)

    estimator = cv_scores['estimator'].values[np.argmax(cv_scores['test_' + scorers[0]].values)]  # select by best test score
    mtrx = cv_scores['Confusion Matrix'].values[np.argmax(cv_scores['test_' + scorers[0]].values)]  # select by best test score
    all_mtrxs = cv_scores['Confusion Matrix'].values
    del cv_scores['estimator']
    del cv_scores['Confusion Matrix']
    if cnd is not None:
        del cv_scores[f'Condition {cnd} Matrix']
    cv_scores_avg = {'avg_' + key: sum(value) / n_splits for key, value in cv_scores.items()}
    cv_scores_med = {'med_' + key: np.median(value) for key, value in cv_scores.items()}
    cv_scores_std = {'std_' + key: value.std() for key, value in cv_scores.items()}
    cv_scores_all = {**cv_scores_avg, **cv_scores_med, **cv_scores_std, **{'all_' + k: str(list(v)).replace('\n','') for k, v in cv_scores.items()}
                     }

    cv_scores_all['Confusion Matrix'] = str(list(map(lambda x: list(map(list, x)), all_mtrxs))) # how ugly is this?? can't get rid of nested array tags from numpy, even tho 3-tensor is structured!
    cv_scores_all['estimator'] = estimator


    if external_validation != 0:
        clf0 = clf
        clf0.fit(X, y)
        ext_pred = clf0.predict(X_ext)
        ext_scores = dict()
        ext_scores['acc'] = accuracy_score(y_ext, ext_pred)
        ext_scores = {**ext_scores, **cnd_score(confusion_matrix(y_ext, ext_pred), cnd, verbose=True)}
        ext_scores = {'extcv_' + key: value for key, value in ext_scores.items()}
        mean_cv_scores = {**mean_cv_scores, **ext_scores}

    if total_train:
        clf0 = clf
        mean_cv_scores = {**mean_cv_scores, **(classifierScore(data, clf0, cnd=cnd, prefix='total_train_'))}
        mean_cv_scores['estimator'] = clf0

    # return {**{'subjects': len(y)}, **mean_cv_scores}
    return {**{'subjects': len(y)}, **cv_scores_all}


# Score a TRAINED classifier given standard scoring input
def classifierScore(data, clf, cnd=None, threshold=None, prefix=''):
    X = data.x_scaled.values
    y = data.y.values.astype('int64').ravel()
    y_pred = clf.predict(X) if threshold is None else threshold_clf(clf, cnd, threshold).predict(X)
    mtrx = confusion_matrix(y, y_pred)
    columns = dict()
    columns['acc'] = accuracy_score(y, y_pred)
    if cnd is not None:
        columns = {**columns, **(cnd_score(mtrx, cnd))}
    if prefix != '':
        columns = {prefix + key:value for key, value in columns.items()}
    return columns


# Given an n x n confusion matrix, collapse to a 2 x 2 confusion matrix for scoring with index :cnd as 'positive'
def cnd_score(mtrx, cnd, verbose=False):
    columns = dict()
    tp = mtrx[cnd][cnd]
    fn = np.sum(mtrx[cnd, :]) - tp
    fp = np.sum(mtrx[:, cnd]) - tp
    tn = np.sum(mtrx) - (tp + fn + fp)

    if verbose:
        columns['tn'] = tn
        columns['fn'] = fn
        columns['fp'] = fp
        columns['tp'] = tp

    columns[f"Condition {cnd} Matrix"] = np.asarray([[tp, fn], [fp, tn]])
    if fn + tp != 0:
        columns['TPR'] = tp / (fn + tp)
        columns['FNR'] = fn / (fn + tp)
    else:
        columns['TPR'] = np.nan
        columns['FNR'] = np.nan
    if fp + tn != 0:
        columns['FPR'] = fp / (fp + tn)
        columns['TNR'] = tn / (fp + tn)
    else:
        columns['FPR'] = np.nan
        columns['TNR'] = np.nan
    if tp + fp != 0:
        columns['precision'] = tp / (tp + fp)
    else:
        columns['precision'] = np.nan
    if columns['TPR'] != np.nan:
        pass
    if columns['precision'] != np.nan:
        if columns['TPR'] + columns['precision'] != 0:
            columns['F1'] = 2 * columns['TPR'] * columns['precision'] / (columns['TPR'] + columns['precision'])
        else:
            columns['F1'] = np.nan

    return columns


'''
Custom Estimators For Classification
'''


# Classes for custom sklearn estimator
class Estimator:
    def __init__(self, **kwargs):
        self.params = {**kwargs}

    def get_params(self, **kwargs):
        return self.params


def apply_threshold(proba_list, cnd, threshold):
    if proba_list[cnd] >= threshold:
        return cnd
    else:
        proba_list[cnd] = 0
        return np.argmax(proba_list)


class threshold_clf(Estimator):
    def __init__(self, clf, cnd, threshold=.5):
        super().__init__(clf=clf, cnd=cnd, threshold=threshold)
        self.clf = clf
        self.fit_clf = None
        self.cnd = cnd
        self.threshold = threshold
        # changeClassTypeName(threshold_clf,
        #                    f'{type(clf).__name__}_t({threshold})')

    def fit(self, X, y, **kwargs):
        if self.fit_clf is None:
            self.fit_clf = self.clf.fit(X, y, **kwargs)  # Needs work
        else:
            self.fit_clf = self.clf.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.fit_clf is None:
            raise Exception('Estimator not fit')
        probas = self.fit_clf.predict_proba(X, **kwargs)
        preds = [apply_threshold(proba, self.cnd, self.threshold) for proba in probas]
        return preds

    def update_threshold(self, threshold):
        self.threshold = threshold


'''
Dataset Transformers
'''


# Remove all elements not 'kept' in some column
def column_filter(cname, keep):
    def ret(df, feature_columns):
        if cname in feature_columns:
            feature_columns.remove(cname)
        return df[df[cname] == keep]
    return ret


def average_transformer(new_cname, cname_regex, series_method="mean"):
    def ret(df, feature_columns):
        cnames = [cname for cname in df.columns.values if re.match(cname_regex, cname)]
        df[new_cname] = df.apply(lambda row: getattr(row[cnames], series_method)(), axis='columns')
    return ret