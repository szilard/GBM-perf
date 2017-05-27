library(data.table)
library(ROCR)
library(lightgbm)
library(parallel)
library(Matrix)

set.seed(123)

d_train <- fread("train-10m.csv")
d_test <- fread("test.csv")


## dealing with cats as suggested by @Laurae2 here https://github.com/szilard/GBM-perf/issues/2#issuecomment-304437441

d_train_test <- rbind(d_train, d_test)
cols_cats <- setdiff(names(which(sapply(d_train, is.character))),"dep_delayed_15min")
for (k in cols_cats) d_train_test[[k]] <- as.numeric(as.factor(d_train_test[[k]]))

n1 <- nrow(d_train)
n2 <- nrow(d_test)
p <- ncol(d_train)-1
X_train <- as.matrix(d_train_test[1:n1,1:p])
X_test <- as.matrix(d_train_test[(n1+1):(n1+n2),1:p])


dlgb_train <- lgb.Dataset(data = X_train, label = ifelse(d_train$dep_delayed_15min=='Y',1,0))


system.time({
md <- lgb.train(data = dlgb_train, objective = "binary", 
            nrounds = 100, num_leaves = 512, learning_rate = 0.1, categorical_feature = cols_cats)
})



system.time({
  phat <- predict(md, data = X_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values


