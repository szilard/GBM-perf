library(data.table)
library(ROCR)
library(catboost)

set.seed(123)

d_train <- fread("train-1m.csv", stringsAsFactors=TRUE)
d_test <- fread("test.csv", stringsAsFactors=FALSE)
d_train_test <- rbind(d_train, d_test)    ## to match factors in train and test
p <- ncol(d_train_test)-1

d_train_test$dep_delayed_15min <- ifelse(d_train_test$dep_delayed_15min=="Y",1,0)   ## need numeric y

d_train <- d_train_test[(1:nrow(d_train)),]
d_test <-  d_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]


dx_train <- catboost.load_pool(d_train[,1:p], label = d_train$dep_delayed_15min)
dx_test  <- catboost.load_pool(d_test[,1:p])


params <- list(iterations = 100, depth = 10, learning_rate = 0.1,
   #max_ctr_complexity=1,
   #one_hot_max_size=255,
   #leaf_estimation_iterations=1,
   thread_count = parallel::detectCores())
system.time({
  md <- catboost.train(learn_pool = dx_train, test_pool = NULL, params = params)
})


system.time({
  phat <- catboost.predict(md, dx_test)
})
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")


