
library(h2o)

h2o.init(nthreads=-1)

dx_train <- h2o.importFile(path = "train-1m.csv")
dx_test <- h2o.importFile(path = "test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]


## xgboost (on GPU/CPU) via h2o deep water

system.time({
  md <- h2o.xgboost(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, distribution = "bernoulli", 
          ntrees = 100, max_depth = 10, learn_rate = 0.1, backend = "gpu")
})


## h2o.auc(h2o.performance(md, dx_test))     ## not working (bug), work around:
phat <- as.data.frame(h2o.predict(md, dx_test))$C3
library(ROCR)
rocr_pred <- prediction(phat, as.data.frame(dx_test)$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values



