import pandas as pd
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import random
import numpy as np

random.seed(123)
np.random.seed(123)

df_train_full = pd.read_csv('data/train-0.1m.csv')
df_test = pd.read_csv('data/test.csv').sample(n=10000)

X_test = df_test.drop(columns=['dep_delayed_15min'])
y_test = df_test['dep_delayed_15min']

sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

for sample_size in sample_sizes:
    df_train = df_train_full.sample(n=sample_size)
    X_train = df_train.drop(columns=['dep_delayed_15min'])
    y_train = df_train['dep_delayed_15min']

    # Combine train and test for encoding
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Encode categorical columns on combined data
    label_encoders = {}
    for col in X_combined.columns:
        if X_combined[col].dtype == 'object':
            le = LabelEncoder()
            X_combined[col] = le.fit_transform(X_combined[col].astype(str))
            label_encoders[col] = le

    # Split back into train and test
    X_train = X_combined.iloc[:len(X_train)]
    X_test_encoded = X_combined.iloc[len(X_train):]

    # Encode target variable
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    # Scale max_depth based on sample size
    # 1000 samples and below -> max_depth=3, 100000 samples -> max_depth=10
    if sample_size <= 1000:
        max_depth = 3
    else:
        max_depth = int(3 + (sample_size - 1000) * (10 - 3) / (100000 - 1000))

    # Train XGBoost model
    start_time = time.time()
    clf = xgb.XGBClassifier(n_jobs=-1, 
            max_depth=max_depth, n_estimators=100, learning_rate=0.1)
    clf.fit(X_train, y_train_encoded)

    # Predict and measure time
    prediction_probabilities = clf.predict_proba(X_test_encoded)
    predict_time = time.time() - start_time

    roc_auc = roc_auc_score(y_test_encoded, prediction_probabilities[:, 1])
    print(f"XGBoost Baseline - Train size: {sample_size:6d} - Run time: {predict_time:.4f}s - Test AUC: {roc_auc:.4f}")
