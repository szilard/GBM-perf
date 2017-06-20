library(data.table)
library(ROCR)
library(xgboost)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- fread("train-1m.csv")
d_test <- fread("test.csv")


system.time({
  X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
  n1 <- nrow(d_train)
  n2 <- nrow(d_test)
  X_train <- X_train_test[1:n1,]
  X_test <- X_train_test[(n1+1):(n1+n2),]
})
dim(X_train)

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))



system.time({
n_proc <- detectCores()
md <- xgb.train(data = dxgb_train, nthread = n_proc, objective = "binary:logistic", 
            nround = 100, eta = 0.1,
            tree_method = "hist", grow_policy = "lossguide",
            max_depth = 0, max_leaves = 512)
})



system.time({
  phat <- predict(md, newdata = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


