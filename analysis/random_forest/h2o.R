library(h2o)

h2o.init()

dx_train <- h2o.importFile("train-1m.csv")
dx_test <- h2o.importFile("test.csv")


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]


system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, 
          ntrees = 100, max_depth = 10, 
          nbins = 100)
})
cat(h2o.auc(h2o.performance(md, dx_test)),"\n")


system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, 
          ntrees = 100, max_depth = 15, 
          nbins = 100)
})
cat(h2o.auc(h2o.performance(md, dx_test)),"\n")


system.time({
  md <- h2o.randomForest(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, 
          ntrees = 100, max_depth = 20, 
          nbins = 100)
})
cat(h2o.auc(h2o.performance(md, dx_test)),"\n")


