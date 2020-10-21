import cudf
import xgboost as xgb

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn import metrics


d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")


d_all = pd.concat([d_train,d_test])

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d_all[col] = preprocessing.LabelEncoder().fit_transform(d_all[col])
  
X_all_cat = preprocessing.OneHotEncoder(categories="auto").fit_transform(d_all[vars_cat])   
X_all = sparse.hstack((X_all_cat, d_all[vars_num])).tocsr()                               
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)

X_train = X_all[0:d_train.shape[0]]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]

#--- RAM 1.3G (0.7GB before ipython kernel)


## ******* CPU

dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   
print(metrics.roc_auc_score(y_test, y_pred))

#Wall time: 8.39 s
#0.7490139621617369

#--- 1.4GB RAM


## ******* data RAM (CPU), train GPU

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'gpu_hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   
print(metrics.roc_auc_score(y_test, y_pred))

#Wall time: 7.89 s
#0.748896157113143

#--- 0.9GB GPU mem / 2.3GB RAM

#--- reset GPU (restart kernel, reload data)


## ******* all GPU (data+train)

#X_train_GPU = cudf.DataFrame.from_pandas(X_train)      ## doesn't work from sparse matrix
#y_train_GPU = cudf.Series(y_train)
#X_test_GPU = cudf.DataFrame.from_pandas(X_test)
#y_test_GPU = cudf.Series(y_test)

#X_train_GPU = cudf.DataFrame.from_pandas(d_train[vars_cat+vars_num])        ## cannot create DMatrix later with cat.vars
#y_train_GPU = cudf.Series(y_train)
#X_test_GPU = cudf.DataFrame.from_pandas(d_test[vars_cat+vars_num])
#y_test_GPU = cudf.Series(y_test)

X_train_GPU = cudf.DataFrame.from_pandas(pd.DataFrame(data=X_train.todense()))    
y_train_GPU = cudf.Series(y_train)
X_test_GPU = cudf.DataFrame.from_pandas(pd.DataFrame(data=X_test.todense()))
y_test_GPU = cudf.Series(y_test)

#--- 7GB GPU mem / 2GB RAM (max during transfer 7GB RAM)


dxgb_train_GPU = xgb.DMatrix(X_train_GPU, label = y_train_GPU)
dxgb_test_GPU = xgb.DMatrix(X_test_GPU)

#--- 12GB GPU mem

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'gpu_hist'}             
%time md = xgb.train(param, dxgb_train_GPU, num_boost_round = 100)

#!!!! crashes kernel (too much GPU mem?)

y_pred = md.predict(dxgb_test_GPU)   
print(metrics.roc_auc_score(y_test, y_pred))



