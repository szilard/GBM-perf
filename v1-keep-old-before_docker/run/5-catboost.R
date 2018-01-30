library(data.table)
library(ROCR)
library(catboost)

set.seed(123)

d_train <- fread("train-1m.csv", stringsAsFactors=TRUE)
d_test <- fread("test.csv", stringsAsFactors=FALSE)
d_train_test <- rbind(d_train, d_test)    ## to match factors in train and test
p <- ncol(d_train_test)-1

d_train_test$DepTime <- as.numeric(d_train_test$DepTime)  ## integer not supported
d_train_test$Distance <- as.numeric(d_train_test$Distance)
d_train_test$dep_delayed_15min <- ifelse(d_train_test$dep_delayed_15min=="Y",1,0)   ## need numeric y

d_train <- d_train_test[(1:nrow(d_train)),]
d_test <-  d_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]


dx_train <- catboost.from_data_frame(d_train[,1:p], target = d_train$dep_delayed_15min)
dx_test  <- catboost.from_data_frame(d_test[,1:p])


params <- list(iterations = 100, depth = 10, learning_rate = 0.1, 
   thread_count = parallel::detectCores())
system.time({
  md <- catboost.train(learn_pool = dx_train, test_pool = NULL, params = params)
})


system.time({
  phat <- catboost.predict(md, dx_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


