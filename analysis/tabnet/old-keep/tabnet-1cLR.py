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

cat_idxs = [ i for i, col in enumerate(X_all.columns) if col in vars_cat]
cat_dims = [ len(np.unique(X_all.iloc[:,i].values)) for i in cat_idxs]

X_train = X_all[0:d_train.shape[0]].to_numpy()
y_train = y_all[0:d_train.shape[0]]
X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])].to_numpy()
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]


MAX_EPOCH = 10 
BS = 1024 

md = TabNetClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=1,
                       ## optimizer_fn=torch.optim.Adam,
                       ## optimizer_params=dict(lr=2e-2),
                       scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                       scheduler_params=dict(max_lr=0.05,
                                             steps_per_epoch=int(X_train.shape[0] / BS),
                                             epochs=MAX_EPOCH,
                                             is_batch_level=True),
                       mask_type='entmax' # "sparsemax"
)

%%time
md.fit( X_train=X_train, y_train=y_train,
    max_epochs=MAX_EPOCH, patience=0,
    ## batch_size=1024, virtual_batch_size=128,
    ## weights=0,
    drop_last = True
)


y_pred = md.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test, y_pred))

