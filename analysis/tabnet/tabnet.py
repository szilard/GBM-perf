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

#normalize
#d_all["DepTime"]=d_all["DepTime"]/2400
#d_all["Distance"]=np.log10(d_all["Distance"]/100)

X_all = d_all[vars_num+vars_cat]
y_all = np.where(d_all["dep_delayed_15min"]=="Y",1,0)

cat_idxs = [ i for i, col in enumerate(X_all.columns) if col in vars_cat]
cat_dims = [ len(np.unique(X_all.iloc[:,i].values)) for i in cat_idxs]

idx_trvl = np.random.choice(["train", "valid"], p =[.8, .2], size=(d_train.shape[0],))

X_train = X_all[0:d_train.shape[0]][idx_trvl=="train"].to_numpy()
y_train = y_all[0:d_train.shape[0]][idx_trvl=="train"]
X_valid = X_all[0:d_train.shape[0]][idx_trvl=="valid"].to_numpy()
y_valid = y_all[0:d_train.shape[0]][idx_trvl=="valid"]

X_test = X_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])].to_numpy()
y_test = y_all[d_train.shape[0]:(d_train.shape[0]+d_test.shape[0])]


md = TabNetClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=1,
                       optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":50, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"
)

%%time
md.fit( X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=1000, patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
)


y_pred = md.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test, y_pred))

