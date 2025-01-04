import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,spearmanr
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet, Ridge
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, \
    roc_curve, auc, f1_score


def base_model():
    b1 = LogisticRegression(penalty='l2', solver='lbfgs', C=0.01, class_weight='balanced')
    b2 = SVC(gamma='scale', C=0.01, kernel='linear', probability=True,random_state=1,class_weight= 'balanced') 
    b3 = BalancedRandomForestClassifier(n_estimators=150, max_depth=6, max_leaf_nodes=10, random_state=1,
                                   criterion='gini', n_jobs=-1, sampling_strategy='auto')
    b4 = MLPClassifier(solver='lbfgs', max_iter=500, alpha=0.01, hidden_layer_sizes=(10,), random_state=1)
    b5 = AdaBoostClassifier(random_state=1, base_estimator=None, learning_rate=0.1, n_estimators=200)

    b6 = GradientBoostingClassifier(random_state=1, n_estimators=200, learning_rate=0.1, max_depth=6)
    b7 = XGBClassifier(booster='gblinear', reg_lambda=0.01)
    b8 = LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
    return b1,b2,b3,b4,b5,b6,b7,b8

def stck_model(model_lst,X_train,y_train):
    total_model_lst = model_lst
    total_model_name = [str(i).strip().replace('\n', '').replace(' ', '') for i in total_model_lst]
    model_dict = {}
    score_dict = {}
    meta_X = list()
    meta_y = list()
    kfold = StratifiedKFold(n_splits=5, shuffle=False)
    for train_ix, test_ix in kfold.split(X_train, y_train):
        fold_f1 = list()
        train_X, test_X = X_train[train_ix], X_train[test_ix]
        train_y, test_y = y_train[train_ix], y_train[test_ix]
        for name_, model in zip(total_model_name, total_model_lst):
            m1=model.fit(train_X, train_y)
            y_valid_pre = m1.predict(test_X1)
            f1 = balanced_accuracy_score(test_y, y_valid_pre)
            fold_f1.append(f1)
        meta_X.append(np.hstack(fold_f1))
    f1_mean = np.mean(np.asarray(meta_X), axis=0)
    for n, name_ in enumerate(total_model_name):
        score_dict[name_] = f1_mean[n]
    cv_result = pd.DataFrame([score_dict]).T.rename(columns={0: 'score'}).sort_values(by='score', ascending=False)
    score_median = cv_result['score'].median()
    index_name = cv_result[cv_result['score'] >= score_median].index.values
    model_name_filterd = list(filter(lambda x: x[0] in index_name, list(zip(total_model_name, total_model_lst))))
    single_model_filtered = [model_name_filterd[i][1] for i in range(len(model_name_filterd))]
    for model in single_model_filtered:
        print('select model:', model.__class__.__name__)
    return model_name_filterd,sf

from sklearn.feature_extraction.text import CountVectorizer
def stck_model2(model_lst,X_train,y_train,X_val,y_val):
    total_model_lst = model_lst
    total_model_name = [str(i).strip().replace('\n', '').replace(' ', '') for i in total_model_lst]
    model_dict = {}
    score_dict = {}
    meta_X = list()
    fold_f1 = list()
    train_X=X_train
    test_X=X_val
    train_y=y_train
    test_y=y_val
    for name_, model in zip(total_model_name, total_model_lst):
        m1=model.fit(train_X, train_y)
        y_valid_pre = m1.predict(test_X)
        y_valid_prob=m1.predict_proba(test_X)[:, 1]
        f1 = balanced_accuracy_score(test_y, y_valid_pre)
        fold_f1.append(f1)
    meta_X.append(np.hstack(fold_f1))
    f1_mean = np.mean(np.asarray(meta_X), axis=0)
    for n, name_ in enumerate(total_model_name):
        score_dict[name_] = f1_mean[n]
    cv_result = pd.DataFrame([score_dict]).T.rename(columns={0: 'score'}).sort_values(by='score', ascending=False)
    score_median = cv_result['score'].median()
    print('Median',score_median)
    index_name = cv_result[cv_result['score'] >= score_median].index.values
    model_name_filterd = list(filter(lambda x: x[0] in index_name, list(zip(total_model_name, total_model_lst))))
    single_model_filtered = [model_name_filterd[i][1] for i in range(len(model_name_filterd))]
    for model in single_model_filtered:
        print('select model:', model.__class__.__name__)
    return model_name_filterd