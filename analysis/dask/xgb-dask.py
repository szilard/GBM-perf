import pandas as pd
from sklearn import metrics

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

import xgboost as xgb


cluster = LocalCluster(n_workers=16, threads_per_worker=1)
client = Client(cluster)

d_train = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/train-1m-intenc.csv")
d_test = pd.read_csv("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/test-1m-intenc.csv")

dx_train = dd.from_pandas(d_train, npartitions=16)
dx_test = dd.from_pandas(d_test, npartitions=1)

X_train = dx_train.iloc[:, :-1].to_dask_array(lengths=True)
y_train = dx_train.iloc[:,-1:].to_dask_array(lengths=True)
X_test = dx_test.iloc[:, :-1].to_dask_array(lengths=True)
y_test = dx_test.iloc[:,-1:].to_dask_array(lengths=True)

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


