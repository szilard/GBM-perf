
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from scipy import sparse
from sklearn import metrics
import h2o4gpu
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


start = time.time()
md = h2o4gpu.GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 10,
      backend = "h2o4gpu", tree_method = "gpu_hist").fit(X_train, y_train)
end = time.time()
print(end - start)

y_pred = md.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, y_pred)



