library(data.table)
library(ROCR)
library(lightgbm)

d_train <- fread("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/train-1m-intenc.csv")
d_test <- fread("https://raw.githubusercontent.com/szilard/benchm-ml--data/master/int_enc/test-1m-intenc.csv")

p <- ncol(d_train)-1
X_train <- as.matrix(d_train[,1:p])
X_test <- as.matrix(d_test[,1:p])
y_train <- d_train$dep_delayed_15min
y_test <- d_test$dep_delayed_15min


dlgb_train <- lgb.Dataset(data = X_train, label = y_train)

system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, num_leaves = 512, learning_rate = 0.1, 
            verbose = 0)
})

phat <- predict(md, data = X_test)
rocr_pred <- prediction(phat, y_test)
performance(rocr_pred, "auc")@y.values[[1]]


## m5.4xlarge 16c (8+8HT)
##    user  system elapsed
## 54.040   0.300   3.848
## 0.7636987
