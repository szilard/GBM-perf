
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import metrics

import xgboost as xgb


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

X_train = X_all[0:d_train.shape[0],]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0]),]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]



md = xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.1, 
          tree_method="hist",  n_jobs=-1)
%time md.fit(X_train, y_train)

y_pred = md.predict_proba(X_test)[:,1]

print(metrics.confusion_matrix(y_test, y_pred>0.7))
print(metrics.roc_auc_score(y_test, y_pred))



dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   

print(metrics.roc_auc_score(y_test, y_pred))

