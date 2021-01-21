import pandas as pd
from sklearn import metrics

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

from lightgbm.dask import DaskLGBMClassifier


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



md = DaskLGBMClassifier(num_leaves=512, learning_rate=0.1, n_estimators=100, tree_learner="data", silent=False)
%time md.fit( client=client, X=X_train, y=y_train)

md_loc = md.to_local()
X_test_loc = X_test.compute()

y_pred = md_loc.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test, y_pred))



