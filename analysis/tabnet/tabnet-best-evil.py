from pytorch_tabnet.tab_model import TabNetClassifier
import torch

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")


d_all = pd.concat([d_train,d_test])

vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
for col in vars_cat:
  d_all[col] = preprocessing.LabelEncoder().fit_transform(d_all[col])

X_all = d_all[vars_num+vars_cat]
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)

X_train = X_all[0:d_train.shape[0]].to_numpy()
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])].to_numpy()
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]


cat_idxs = [ i for i, col in enumerate(X_all.columns) if col in vars_cat]
cat_dims = [ len(np.unique(X_all.iloc[:,i].values)) for i in cat_idxs]
cat_emb_dim = np.floor(np.log(cat_dims)).astype(int)


md = TabNetClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=cat_emb_dim,
                       ## optimizer_fn=torch.optim.Adam,
                       ## optimizer_params=dict(lr=2e-2),
                       ## mask_type='sparsemax',
                       n_steps=1,
)

%%time
md.fit( X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test_EVIL'],
    eval_metric=['auc'],
    max_epochs=100, patience=100,
    ## batch_size=1024, virtual_batch_size=128,
    ## weights=0,
)

y_pred = md.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test, y_pred))

