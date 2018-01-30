
import xgboost as xgb
import pandas as pd
from sklearn import metrics
import time

dtrain = xgb.DMatrix('train-10m.libsvm')
dtest = xgb.DMatrix('test-10m.libsvm')

#params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic'}
#params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'updater':'grow_colmaker'}
#params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'updater':'grow_fast_histmaker'}
#params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'updater':'grow_gpu'}
#params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'updater':'grow_gpu_hist'}

params = {'max_depth':10, 'eta':0.1, 'objective':'binary:logistic', 'tree_method':'gpu_hist'}


num_round = 100

start = time.time()
md = xgb.train(params, dtrain, num_round)
end = time.time()
print(end - start)


phat = md.predict(dtest)
ytest = pd.read_csv("test.csv")['dep_delayed_15min']=='Y'
metrics.roc_auc_score(ytest, phat)



