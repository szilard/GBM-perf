library(data.table)
library(ROCR)
library(xgboost)

d_train <- fread("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/train-1m-intenc.csv")
d_test <- fread("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/test-1m-intenc.csv")

p <- ncol(d_train)-1
X_train <- as.matrix(d_train[,1:p])
X_test <- as.matrix(d_test[,1:p])
y_train <- d_train$dep_delayed_15min
y_test <- d_test$dep_delayed_15min


dxgb_train <- xgb.DMatrix(data = X_train, label = y_train)

system.time({
  md <- xgb.train(data = dxgb_train, 
            objective = "binary:logistic", 
            nround = 100, max_depth = 10, eta = 0.1, 
            tree_method = "hist")
})

phat <- predict(md, newdata = X_test)
rocr_pred <- prediction(phat, y_test)
performance(rocr_pred, "auc")@y.values[[1]]


## m5.4xlarge 16c (8+8HT)
##    user  system elapsed
## 55.630   0.094   3.923
## 0.7527782
