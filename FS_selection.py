import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, RFECV,RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, SMOTENC, KMeansSMOTE, RandomOverSampler
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, \
    roc_curve, auc, f1_score

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = np.expand_dims(ranks, axis=1)
    return ranks


fs=pd.read_excel('./EnLasso_fold{}.xlsx'.format(fold),sheet_name='lasso',skiprows=None)
fs1 = fs.sort_values(ascending=False,by='weight')
fs1=np.array(fs1)
feat_labels=fs1[:500,0] 
f_name = pd.read_excel('./dataset.xlsx',sheet_name='Sheet1',skiprows=None)
fall_labels = np.array(f_name)[1:,0]
index=FS_index(feat_labels,fall_labels)
data_row = np.load('./train_fold.npy'.format(fold))
data = np.array(data_row)
data_x = data[:,1:]
data_x =data_x[:, index]
data_y = data[:, 0]

#######
algorithms = ['Lasso']
fs_name =np.expand_dims(feat_labels, axis=1)
fs_indices1 =np.expand_dims(feat_labels, axis=1)
fs_indices2 =np.expand_dims(feat_labels, axis=1)
fs_indices3 =np.expand_dims(feat_labels, axis=1)
fs_indices4 =np.expand_dims(feat_labels, axis=1)
fs_1 =np.expand_dims(feat_labels, axis=1)
fs_2 =np.expand_dims(feat_labels, axis=1)
fs_3 =np.expand_dims(feat_labels, axis=1)
fs_4 =np.expand_dims(feat_labels, axis=1)

Neg=np.where(data_y==0)
data_Neg=data_x[Neg]
data_Neg_y=data_y[Neg]
i=0
rs=1
#####
for j in range(100):
    ss = StratifiedShuffleSplit(n_splits=2,test_size=0.5,train_size=0.5,random_state=None)
    for train_index, test_index in ss.split(data_x, data_y):
        i = i + 1
        X_train= data_x[train_index]
        y_train=data_y[train_index]
        rus=SVMSMOTE()
        X_train1, y_train1 = rus.fit_resample(data_x, data_y)
        for name in algorithms:
            print("#" * 30)
            print(name)
            if name == 'Lasso':
                clf_lasso = Lasso(alpha=0.01).fit(X_train1, y_train1)
                importances = np.abs(clf_lasso.coef_)
                ranks=rank_to_dict(importances, feat_labels)
                fs_1 = np.concatenate((fs_1, ranks), axis=-1)
                indices_a1 = np.zeros((len(feat_labels), 1))
                ind = np.argsort(importances)[::-1]
                indices_a1[ind, :] = np.arange(1, len(ind) + 1)[..., np.newaxis]
                fs_indices1 = np.concatenate((fs_indices1, indices_a1), axis=-1)
print("i=:",i)
f11 =np.expand_dims(np.mean(fs_1[:,1:],axis=1), axis=1)
fs_m1 = np.concatenate((fs_name,f11), axis=-1)

log_outfile = 'EnLasso_fold{}.xlsx'.format(fold)
writer = pd.ExcelWriter(log_outfile)
fs_m1=pd.DataFrame(fs_m1,columns=['name','weight'])
fs_m1.to_excel(writer,sheet_name='lasso',index=False)
writer.save()
writer.close()