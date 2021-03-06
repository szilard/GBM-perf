library(data.table)
library(ROCR)
library(xgboost)
library(Matrix)


d_train <- fread("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
d_test <- fread("https://s3.amazonaws.com/benchm-ml--main/test.csv")


X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
n1 <- nrow(d_train)
n2 <- nrow(d_test)
X_train <- X_train_test[1:n1,]
X_test <- X_train_test[(n1+1):(n1+n2),]

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))
dxgb_test  <- xgb.DMatrix(data = X_test, label = ifelse(d_test$dep_delayed_15min=='Y',1,0))


system.time({
  md <- xgb.train(data = dxgb_train, 
            objective = "binary:logistic", 
            nround = 1000, max_depth = 10, eta = 0.1, 
            tree_method = "hist",
            early_stopping_rounds = 10, watchlist = list(train=dxgb_train, test_EVIL=dxgb_test), eval_metric = "auc",  
            verbose = 1)
})


phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
cat(performance(rocr_pred, "auc")@y.values[[1]],"\n")


