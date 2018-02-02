
import pandas as pd
import numpy as np
from sklearn import metrics
import catboost 
import multiprocessing


d_train = pd.read_csv("train-1m.csv")
d_test = pd.read_csv("test.csv")

X_train = d_train.drop(['dep_delayed_15min'], axis=1)
X_test = d_test.drop(['dep_delayed_15min'], axis=1)
y_train = np.where(d_train["dep_delayed_15min"]=="Y",1,0)           
y_test = np.where(d_test["dep_delayed_15min"]=="Y",1,0)           

cat_cols = np.where(X_train.dtypes == np.object)[0]


md = catboost.CatBoostClassifier(iterations = 100, depth = 10, learning_rate = 0.1,
               task_type = "GPU")
      ##         thread_count = multiprocessing.cpu_count())
%time md.fit(X_train, y_train, cat_features = cat_cols)


y_pred = md.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, y_pred)






