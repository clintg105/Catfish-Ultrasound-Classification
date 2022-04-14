import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from SciKit_ML import classifiers, classifiers_short

from common_functions import append_dict, partition
from feature_selection.feature_selection import select_and_run_targs

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
# explicitly require experimental features
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 500)

data_dir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\\'
results_dir=data_dir+'Results\\'
fnames = [f.split("\\")[-1] for f in glob.glob(data_dir+"*.csv")]

desiredMaxDim = 2000

for file in fnames:
    drType = "CNN" if ("CNN_" in file) else "stat"

    df = pd.read_csv(data_dir + file)
    nvars = len(df.columns) - 2
    features = ['v' + str(n) for n in range(1, nvars + 1)]
    maxDim = min(desiredMaxDim, nvars)

    drName = file.split("_")[-1][:-4]
    print(drName)

    models = [
        GaussianNB(),
        MLPClassifier(hidden_layer_sizes=(100), activation="logistic", max_iter=300),
        LogisticRegression(penalty="elasticnet", C=0.01, solver="saga", l1_ratio=0.2, max_iter=400),
        KNeighborsClassifier(n_neighbors=5),
        ExtraTreeClassifier(),
        RandomForestClassifier(),
        LinearSVC(),
        AdaBoostClassifier(n_estimators=100),
        GradientBoostingClassifier(),
        VotingClassifier(estimators=[
            ('mlp', MLPClassifier(hidden_layer_sizes=(100), activation="logistic", max_iter=300)),
            ('svc', LinearSVC()),
            ('rfc', RandomForestClassifier())
        ], voting='hard')
    ]

    results = select_and_run_targs(
        df, features, ['class2'], models,
        upto=list(range(100, maxDim + 1, 100)) + [nvars], graph_r2=False, eval_type='c', show_best_runs=3, cv_params=dict(cnd=1), n_cores=-1
    )
    results.to_csv(results_dir + "Results_2Class_" + drType + "_" + drName + ".csv")

    results = select_and_run_targs(
        df, features, ['class5'], models,
        upto=list(range(100, maxDim + 1, 100)) + [nvars], graph_r2=False, eval_type='c', show_best_runs=3, n_cores=-1
    )
    results.to_csv(results_dir+"Results_5Class_"+drType+"_"+drName+".csv")

# df = pd.read_csv(data_dir + 'CorrectedImgsReduced_PrincipalComponentsAnalysis.csv')
#
# targets = ['class5','class2']
# model = classifiers_short
# features = ['v' + str(n) for n in range(1, 800 + 1)]
# # seperate targets for 5 class vs 2 class - so that 2 class can have binary classification ouputs
# results = select_and_run_targs(df, features, ['class5'], model, upto=5, graph_r2=True, eval_type='c', show_best_runs=8, n_cores=-1)
# results = select_and_run_targs(df, features, ['class2'], model, upto=5, graph_r2=True, eval_type='c', show_best_runs=8, cv_params=dict(cnd=0), n_cores=-1)