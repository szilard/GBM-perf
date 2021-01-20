import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics

import xgboost as xgb


d_train = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/train-1m-intenc.csv")
d_test = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/test-1m-intenc.csv")

X_train = d_train.iloc[:, :-1].to_numpy()
y_train = d_train.iloc[:,-1:].to_numpy()
X_test = d_test.iloc[:, :-1].to_numpy()
y_test = d_test.iloc[:,-1:].to_numpy()


dxgb_train = xgb.DMatrix(X_train, label = y_train)
dxgb_test = xgb.DMatrix(X_test)

param = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'hist'}             
%time md = xgb.train(param, dxgb_train, num_boost_round = 100)

y_pred = md.predict(dxgb_test)   
print(metrics.roc_auc_score(y_test, y_pred))


## m5.4xlarge 16c (8+8HT)
## Wall time: 3.3 s
## 0.7527781837199401

