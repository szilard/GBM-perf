library(data.table)
library(ROCR)
library(tabnet)
library(Matrix)


d_train <- fread("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv", stringsAsFactors=TRUE)
d_test <- fread("https://s3.amazonaws.com/benchm-ml--main/test.csv")

## align cat. values (factors)
d_train_test <- rbind(d_train, d_test)
n1 <- nrow(d_train)
n2 <- nrow(d_test)
d_train <- d_train_test[1:n1,]
d_test <- d_train_test[(n1+1):(n1+n2),]


system.time({
  md <- tabnet_fit(dep_delayed_15min ~ . ,d_train, epochs = 10, verbose = TRUE)
})


phat <- predict(md, d_test, type = "prob")$.pred_Y
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]


