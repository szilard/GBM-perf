import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics


d_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")

d_train["dep_delayed_15min"] = np.where(d_train["dep_delayed_15min"]=="Y",1,0)
d_test["dep_delayed_15min"] = np.where(d_test["dep_delayed_15min"]=="Y",1,0)


dtf_train = tfdf.keras.pd_dataframe_to_tf_dataset(d_train, label="dep_delayed_15min")
dtf_test = tfdf.keras.pd_dataframe_to_tf_dataset(d_test, label="dep_delayed_15min")


md = tfdf.keras.GradientBoostedTreesModel(max_depth=10, num_trees=100, shrinkage=0.1)
%time md.fit(x=dtf_train)

y_pred = md.predict(dtf_test)   
print(metrics.roc_auc_score(d_test["dep_delayed_15min"], y_pred))


