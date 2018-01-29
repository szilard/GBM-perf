suppressMessages({
library(data.table)
library(ROCR)
library(xgboost)
library(Matrix)
})

set.seed(123)

d_train <- fread("train.csv", showProgress=FALSE)
d_test <- fread("test.csv", showProgress=FALSE)


X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
n1 <- nrow(d_train)
n2 <- nrow(d_test)
X_train <- X_train_test[1:n1,]
X_test <- X_train_test[(n1+1):(n1+n2),]

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))


cat(system.time({
  md <- xgb.train(data = dxgb_train, 
            objective = "binary:logistic", 
            nround = 100, max_depth = 10, eta = 0.1, 
            tree_method = "gpu_hist")
})[[3]]," ",sep="")


phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
cat(performance(rocr_pred, "auc")@y.values[[1]],"\n")


