import cudf
import xgboost as xgb

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")


d_all = pd.concat([d_train,d_test])

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d_all[col] = preprocessing.LabelEncoder().fit_transform(d_all[col])

X_all = d_all[vars_num+vars_cat]
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)

X_train = X_all[0:d_train.shape[0]]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]

#--- RAM 1.1GB (0.7GB before python kernel)


## ******* CPU

dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   
print(metrics.roc_auc_score(y_test, y_pred))

#Wall time: 5.83 s
#0.7523232418031532

#--- 1.2GB RAM


## ******* data RAM (CPU), train GPU

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'gpu_hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   
print(metrics.roc_auc_score(y_test, y_pred))

#Wall time: 4.57 s    (1st time more because of copy to GPU)
#Wall time: 3.06 s     
#0.7527066683633643

#--- 0.85GB GPU mem

#--- reset GPU (restart kernel, reload data)


## ******* all GPU (data+train)

X_train_GPU = cudf.DataFrame.from_pandas(X_train)
y_train_GPU = cudf.Series(y_train)
X_test_GPU = cudf.DataFrame.from_pandas(X_test)
y_test_GPU = cudf.Series(y_test)

#--- 0.8 GPU mem  / 2GB RAM 

dxgb_train_GPU = xgb.DMatrix(X_train_GPU, label = y_train_GPU)
dxgb_test_GPU = xgb.DMatrix(X_test_GPU)

#--- 0.9GB GPU mem

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'gpu_hist'}             
%time md = xgb.train(param, dxgb_train_GPU, num_boost_round = 100)

y_pred = md.predict(dxgb_test_GPU)   
print(metrics.roc_auc_score(y_test, y_pred))

#Wall time: 3.16 s
#0.7527066683633643

#--- 1GB GPU mem



#type(X_train)      # pd DF
#type(y_train)      # np array
#type(X_train_GPU)  # cuDF
#type(y_train_GPU)  # cuSer


