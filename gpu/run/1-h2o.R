suppressMessages({
library(h2o)
})

sink("/dev/null") 
h2o.init()
sink()
h2o.no_progress()

dx_train <- h2o.importFile("train.csv")
dx_test <- h2o.importFile("test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]


cat(system.time({
  md <- h2o.xgboost(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, 
          distribution = "bernoulli", 
          ntrees = 100, max_depth = 10, learn_rate = 0.1, 
          backend = "gpu")
})[[3]]," ",sep="")


cat(h2o.auc(h2o.performance(md, dx_test)),"\n")


