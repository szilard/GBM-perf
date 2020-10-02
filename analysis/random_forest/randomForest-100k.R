library(data.table)
library(ROCR)
library(randomForest)
library(parallel)

set.seed(123)

d_train <- fread("train-0.1m.csv")
d_test <- fread("test.csv")

## "Can not handle categorical predictors with more than 53 categories."
X_train_test <-  model.matrix(dep_delayed_15min ~ ., data = rbind(d_train, d_test))
X_train <- X_train_test[1:nrow(d_train),]
X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]

gc()
## RAM 2.9GB


system.time({
md <- randomForest(X_train, as.factor(d_train$dep_delayed_15min), ntree = 1)
})
## 4.5GB / max 4.7GB
## 44sec

gc()
## 2.9GB


system.time({
md <- randomForest(X_train, as.factor(d_train$dep_delayed_15min), ntree = 2)
})
## 4.5GB / max 4.7GB
## 81sec

gc()
## 2.9GB


system.time({
mds <- mclapply(1:2,
        function(x) randomForest(X_train, as.factor(d_train$dep_delayed_15min), ntree = 1), 
        mc.cores = 2)
md <- do.call("combine", mds)
})
## 2.9GB (max 6.6GB)
## 48sec

gc()
## 2.9GB



## too slow to run:

system.time({
n_proc <- detectCores()/2
mds <- mclapply(1:n_proc,
        function(x) randomForest(X_train, as.factor(d_train$dep_delayed_15min), ntree = floor(1/n_proc)), 
        mc.cores = n_proc)
md <- do.call("combine", mds)
})


system.time({
phat <- predict(md, newdata = X_test, type = "prob")[,"Y"]
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]



