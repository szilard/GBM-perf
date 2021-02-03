
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import metrics
import xgboost as xgb
import time


d_train = pd.read_csv("train.csv")
d_test = pd.read_csv("test.csv")
d = pd.concat([d_train, d_test])


vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d[col] = preprocessing.LabelEncoder().fit_transform(d[col])
 
 
X_cat = preprocessing.OneHotEncoder().fit_transform(d[vars_cat])    
X = sparse.hstack((X_cat, d[vars_num]), format = "csr")                            
      
y = np.where(d["dep_delayed_15min"]=="Y",1,0)                     


X_train = X[0:d_train.shape[0],:]
X_test = X[d_train.shape[0]:,:]

y_train = y[0:d_train.shape[0]]
y_test = y[d_train.shape[0]:]


dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 
             'tree_method':'gpu_hist',
             'silent':1}

start = time.time()
md = xgb.train(param, dxgb_train, num_boost_round = 100)
end = time.time()
print(end - start)

y_pred = md.predict(dxgb_test)   
metrics.roc_auc_score(y_test, y_pred)


