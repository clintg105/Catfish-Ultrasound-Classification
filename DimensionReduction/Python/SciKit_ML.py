# https://scikit-learn.org/stable/supervised_learning.html
# CLASSIFICATION
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
# explicitly require experimental features
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier



classifiers = [
    # Glossary:
    # OUp: (assumed) Online Updating via method partial_fit(X, y)
    # Coe: (assumed) Weights assigned to the features via attribute coef_
    # Prb: (assumed) Probabilities available via method predict_proba(X); (df) tag indicates method decision_function
    # CPb: Class Probabilities available


    # 'SIMPLE' METHODS
    DummyClassifier(),
    KNeighborsClassifier(n_neighbors=50),
    BaggingClassifier(base_estimator=DecisionTreeClassifier()),  # fits base classifier on random subsets of dataset and aggregate predictions (!Coe, !OUp)
    CalibratedClassifierCV(base_estimator=DecisionTreeClassifier()),  # calibrate base_estimator to provide more accurate Prb outputs (!Coe, !OUp) (uses decision_function method or predict_proba)


    # METHOD: Decision tree
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(n_estimators=10),  # Found very useful (could easily overfit!)


    # METHOD: Support Vector Machines (https://scikit-learn.org/stable/modules/svm.html)
    # SVC(probability=True),  # (Slow, !OUp)
    # NuSVC(),  # SVC with control for num of support vectors (!Prb (df), !OUp)
    LinearSVC(),  # SVC(kernel=’linear’) but faster on large datasets (!Prb (df), !OUp)


    # METHOD: Discriminant Analysis (https://scikit-learn.org/stable/modules/lda_qda.html)
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),


    # METHOD: Labels
    LabelPropagation(),  # Unregularized graph based semi-supervised learning (!Coe, !OUp)
    LabelSpreading(),  # LabelPropagation normalized for noise (!Coe, !OUp)


    # METHODS: Statistical Learning -----------------------------------------------------------------------------------

    # METHOD: Bayesian
    GaussianNB(),  # (!Coe, CPb via class_prior_)
    # MultinomialNB(),  # NB for discrete features (REQUIRES VALS >0) (CPb via class_log_prior_, P(x_i|y) via feature_log_prob_, Coe supported as linear interpretation)
    # ComplementNB(),  # MultinomialNB for unbalanced datasets (REQUIRES VALS >0) (!Coe, !CPb)

    # METHOD: Gaussian Clustering
    GaussianMixture(),  # (Coe via weights_, other attribs: means_, covariances_, precisions_, lower_bound_ (on the log-likelihood of the best fit of EM))
    BayesianGaussianMixture(),  # same as GaussianMixture, may approximate posterior distribution via *_prior_ attribs

    # METHOD: Stochastic Processes
    GaussianProcessClassifier(),  # based on Laplace approximation (!Coe, !OUp)


    # METHODS: ML Optimizers ------------------------------------------------------------------------------------------

    # METHOD: Gradient Descent
    SGDClassifier(loss='modified_huber'),  # loss options are 'modified_huber' (outlier tolerance & Prb), 'hinge' (linear SVM), 'squared_hinge', 'log' (LogisticRegression), 'perceptron' + regression options such as ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, ‘squared_epsilon_insensitive’
    LogisticRegression(),  # like SGDClassifier (special case), supports ‘l1’, ‘l2’ & ‘elasticnet’ penalties (!OUp)
    MLPClassifier(hidden_layer_sizes=(10, 5), activation="logistic", max_iter=800),

    # METHOD: Boosting (to focus on more difficult cases) (!OUp, Coe via feature_importances_)
    AdaBoostClassifier(),
    GradientBoostingClassifier(),  # may use any C1 loss, predicts w/ regression trees
    HistGradientBoostingClassifier(),  # fast GradientBoostingClassifier (Experimental, !Coe)


    # Appendix --------------------------------------------------------------------------------------------------------
    # Multiclass Algorithms: https://scikit-learn.org/stable/modules/multiclass.html
    ]

classifiers_short = [
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(20), activation="logistic", max_iter=300),
    LogisticRegression(penalty="elasticnet",C=0.01,solver="saga",l1_ratio=0.2,max_iter=400),
    KNeighborsClassifier(n_neighbors=2)
    ]