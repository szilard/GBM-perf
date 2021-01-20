import pandas as pd
from sklearn import metrics

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
from dask_ml import preprocessing

import xgboost as xgb


cluster = LocalCluster(n_workers=16, threads_per_worker=1)
client = Client(cluster)

d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")
d_all = pd.concat([d_train,d_test])

dx_all = dd.from_pandas(d_all, npartitions=16)

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  dx_all[col] = preprocessing.LabelEncoder().fit_transform(dx_all[col])
  
X_all = dx_all[vars_cat+vars_num].to_dask_array(lengths=True)      
y_all = da.where((dx_all["dep_delayed_15min"]=="Y").to_dask_array(lengths=True),1,0)  

X_train = X_all[0:d_train.shape[0],]
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0]),]
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]

X_train.persist()
y_train.persist()

client.has_what()


dxgb_train = xgb.dask.DaskDMatrix(client, X_train, y_train)
dxgb_test = xgb.dask.DaskDMatrix(client, X_test)


param = {'objective':'binary:logistic', 'tree_method':'hist', 'max_depth':10, 'eta':0.1}             
%time md = xgb.dask.train(client, param, dxgb_train, num_boost_round = 100)


y_pred = xgb.dask.predict(client, md, dxgb_test)
y_pred_loc = y_pred.compute()
y_test_loc = y_test.compute()
print(metrics.roc_auc_score(y_test_loc, y_pred_loc))


## m5.4xlarge 16c (8+8HT)
## Wall time: 20.5 s
## 0.7958538649110775

