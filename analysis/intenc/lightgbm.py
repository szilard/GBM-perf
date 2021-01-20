import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics

import lightgbm as lgb


d_train = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/train-1m-intenc.csv")
d_test = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/test-1m-intenc.csv")

X_train = d_train.iloc[:, :-1].to_numpy()
y_train = d_train.iloc[:,-1:].to_numpy()
X_test = d_test.iloc[:, :-1].to_numpy()
y_test = d_test.iloc[:,-1:].to_numpy()


md = lgb.LGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=100)
%time md.fit(X_train, y_train)


y_pred = md.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test, y_pred))


## m5.4xlarge 16c (8+8HT)
## Wall time: 3.77 s
## 0.7636986921602019


