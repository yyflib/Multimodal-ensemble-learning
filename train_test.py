import numpy as np
import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, RFECV
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA,NMF,SparsePCA,KernelPCA
from sklearn import cluster
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold, KFold,train_test_split,RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, SMOTENC, KMeansSMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, NearMiss, EditedNearestNeighbours, NeighbourhoodCleaningRule,RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.base import BaseSampler
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier
from collections import Counter
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
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, \
    roc_curve, auc, f1_score
import shap
from shap.maskers import Independent
warnings.filterwarnings('ignore')
from code_lib import *


log_cols_test = ["bacc_mean","bac_std","AUC_mean","AUC_std", "Spe_mean","Spe_std","Sen_mean", "Sen_std","f1_mean","f1_std"]
log_cols = [ "test_auc", "test_auc(std)","f_meaure","f_meaure(std)","pca","pca(std)"]
#######
i = -1
best_a=[]
LR0,SVC0,BRF0,MLP0,Ada0,GBDT0,XGB0,LGB0=base_model()
model_lst = [LR0,SVC0,BRF0,MLP0,Ada0,GBDT0,XGB0,LGB0]
for ff in range(1,6):
    fold=ff
    #########
    fs=pd.read_excel('./Lasso_fold{}.xlsx'.format(fold),sheet_name='lasso',skiprows=None)
    fs1 = fs.sort_values(ascending=False,by='weight')
    fs1=np.array(fs1)
    fe_name=fs1[:,0]
    fe_name_weight=fs1[:,1]
    ####load dataset
    data_train = pd.read_excel('fold{}(train).xlsx'.format(fold))
    feat_label = data_train.columns.values
    fe_label = feat_label[1:]
    data_tr = np.array(data_train)
    index=FS_index(fe_name,fe_label)
    data_tr_x =data_tr[:,1:]
    Train_data = data_tr_x[:, index]
    Train_label = data_tr[:, 0]
#########
    data_val = pd.read_excel('./fold{}(val).xlsx'.format(fold))
    data_val = np.array(data_val)
    data_val_x =data_val[:,1:]
    Val_data= data_val_x[:, index]
    Val_label = data_val[:, 0]
#########
    data_test = pd.read_excel('./fold{}(test).xlsx'.format(fold))
    data_te = np.array(data_test)
    data_te_x =data_te[:,1:]
    Test_data= data_te_x[:, index]
    Test_label = data_te[:, 0]
    ################################################################################
    ########################training 
    idx=np.where(fe_name_weight!=0)
    l=len(fe_name[idx])
    n=l-1
    pca_f = []
    bacc_val, auc_val, sen_val, spe_val, gmean_val, f1_val = [], [], [], [], [], []
    bacc_test,auc_test,sen_test,spe_test,gmean_test,f1_test = [],[],[],[],[],[]
    log_outfile1 =result_path1+ 'ROC curve/'+result_path2+'_fold{}.xlsx'.format(fold)
    writer1 = pd.ExcelWriter(log_outfile1)
    for ff in range(n,l):
        i = i + 1
        print("i=", i)
        Train_data1=Train_data[:, :ff]
        x_val1 = Val_data[:, :ff]
        y_val1 = Val_label
        x_test1= Test_data[:, :ff]
        y_test1= Test_label
        rus = SVMSMOTE(random_state=None)
        x_train2, y_train2 = rus.fit_resample(Train_data1, Train_label)
        print('Count:',Counter(y_train2))
        sf = PLS(n_components=s).fit(x_train2, y_train2)
        x_train3 = sf.transform(x_train2)
        print('PLS_:',x_train3.shape)
        f_num=x_train3.shape[1]
        pca_f.append(f_num)
        y_train3 = y_train2
        x_val=sf.transform(x_val1)
        y_val=y_val1
        x_test=sf.transform(x_test1)
        y_test=y_test1
        model_name_filterd = stck_model2(model_lst, x_train3, y_train3,x_val,y_val)
        est = StackingClassifier(estimators=model_name_filterd,
                                  final_estimator= LogisticRegression(penalty='l2', solver='lbfgs', C=0.01, class_weight='balanced'))
        algo_all=est.fit(x_train3, y_train3)

############val
        val_pred= algo_all.predict(x_val)
        val_prob = algo_all.predict_proba(x_val)[:, 1]
        bacc_val1,sens_val1,spe_val1,gmean_val1,f1_val1,auc_val1=classfy_results(y_val,val_pred,val_prob)
        print("balacc_val: {:.4%}".format(bacc_val1))
        bacc_val.append(bacc_val1 * 100)
        sen_val.append(sens_val1 * 100)
        spe_val.append(spe_val1 * 100)
        gmean_val.append(gmean_val1 * 100)
        f1_val.append(f1_val1 * 100)
        auc_val.append(auc_val1)
############test
        test_pred = algo_all.predict(x_test)
        test_prob = algo_all.predict_proba(x_test)[:, 1]
        bacc_test1, sens_test1, spe_test1, gmean_test1, f1_test1, auc_test1 = classfy_results(y_test, test_pred, test_prob)
        print("balacc_test: {:.4%}".format(bacc_test1))
        bacc_test.append(bacc_test1*100)
        sen_test.append(sens_test1 * 100)
        spe_test.append(spe_test1*100)
        gmean_test.append(gmean_test1*100)
        f1_test.append(f1_test1*100)
        auc_test.append(auc_test1)
        y_test = np.array(y_test)[..., np.newaxis]
        test_prob2= np.array(test_prob)[..., np.newaxis]
        test_pred2= np.array(test_pred)[..., np.newaxis]
        results = np.concatenate((y_test, test_prob2), axis=-1)
        results = np.concatenate((results, test_pred2), axis=-1)
        results1 = pd.DataFrame(data=results, dtype='float')
        results1.to_excel(writer1,sheet_name='f{}'.format(ff), index=False)
    writer1.save()
    writer1.close()
############val results
    pca_a    =np.array(pca_f)[..., np.newaxis]
    val_bac =np.array(bacc_val)[..., np.newaxis]
    val_spe =np.array(spe_val)[..., np.newaxis]
    val_sen = np.array(sen_val)[..., np.newaxis]
    val_gmean = np.array(gmean_val)[..., np.newaxis]
    val_f1  = np.array(f1_val)[..., np.newaxis]
    val_auc = np.array(auc_val)[..., np.newaxis]
    bac_t = np.concatenate((pca_a, val_bac), axis=-1)
    bac_t = np.concatenate((bac_t, val_sen),axis=-1)
    bac_t = np.concatenate((bac_t, val_spe),axis=-1)
    bac_t = np.concatenate((bac_t, val_gmean), axis=-1)
    bac_t = np.concatenate((bac_t, val_f1),axis=-1)
    bac_t = np.concatenate((bac_t, val_auc), axis=-1)
############test results
    test_bac = np.array(bacc_test)[..., np.newaxis]
    test_sen = np.array(sen_test)[..., np.newaxis]
    test_spe = np.array(spe_test)[..., np.newaxis]
    test_gmean = np.array(gmean_test)[..., np.newaxis]
    test_f1 = np.array(f1_test)[..., np.newaxis]
    test_auc = np.array(auc_test)[..., np.newaxis]
    bac_t = np.concatenate((bac_t, test_bac), axis=-1)
    bac_t = np.concatenate((bac_t, test_sen), axis=-1)
    bac_t = np.concatenate((bac_t, test_spe), axis=-1)
    bac_t = np.concatenate((bac_t, test_gmean), axis=-1)
    bac_t = np.concatenate((bac_t, test_f1), axis=-1)
    bac_t = np.concatenate((bac_t, test_auc), axis=-1)
