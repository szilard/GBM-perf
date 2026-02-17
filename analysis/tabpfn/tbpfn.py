from tabpfn import TabPFNClassifier
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
import random
import numpy as np

random.seed(123)
np.random.seed(123)

df_train_full = pd.read_csv('data/train-0.1m.csv')
df_test = pd.read_csv('data/test.csv').sample(n=10000)

X_test = df_test.drop(columns=['dep_delayed_15min']).values  
y_test = df_test['dep_delayed_15min'].values

sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

for sample_size in sample_sizes:
    df_train = df_train_full.sample(n=sample_size)
    X_train = df_train.drop(columns=['dep_delayed_15min']).values  
    y_train = df_train['dep_delayed_15min'].values

    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)

    start_time = time.time()
    prediction_probabilities = clf.predict_proba(X_test)
    predict_time = time.time() - start_time

    roc_auc = roc_auc_score(y_test, prediction_probabilities[:, 1])
    print(f"Train size: {sample_size:5d} - Run time: {predict_time:.4f}s - Test AUC: {roc_auc:.4f}")

