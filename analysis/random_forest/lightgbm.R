suppressMessages({
library(data.table)
library(ROCR)
library(lightgbm)
library(Matrix)
})

set.seed(123)

d_train <- fread("train-1m.csv", showProgress=FALSE)
d_test <- fread("test.csv", showProgress=FALSE)

d_all <- rbind(d_train, d_test)
d_all$dep_delayed_15min <- ifelse(d_all$dep_delayed_15min=="Y",1,0)

d_all_wrules <- lgb.convert_with_rules(d_all)       
d_all <- d_all_wrules$data
cols_cats <- names(d_all_wrules$rules) 

d_train <- d_all[1:nrow(d_train)]
d_test <- d_all[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test))]

p <- ncol(d_all)-1
dlgb_train <- lgb.Dataset(data = as.matrix(d_train[,1:p]), label = d_train$dep_delayed_15min)

auc <- function() {
  phat <- predict(md, data = as.matrix(d_test[,1:p]))
  rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
  cat(performance(rocr_pred, "auc")@y.values[[1]],"\n")
}


## GBM num_leaves vs max_depth

system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, num_leaves = 512, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, num_leaves = 2**10, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 10, num_leaves = 2**17, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()




## GBM vs RF

system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, num_leaves = 512, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 10, num_leaves = 2**17, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 10, num_leaves = 2**17, 
            boosting_type = "rf", bagging_freq = 1, bagging_fraction = 0.632, feature_fraction = 1/sqrt(p),
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 15, num_leaves = 2**17, 
            boosting_type = "rf", bagging_freq = 1, bagging_fraction = 0.632, feature_fraction = 1/sqrt(p),
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()


system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 20, num_leaves = 2**17, 
            boosting_type = "rf", bagging_freq = 1, bagging_fraction = 0.632, feature_fraction = 1/sqrt(p),
            categorical_feature = cols_cats,
            verbose = 2)
})
auc()



## GBM deep

system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, max_depth = 20, num_leaves = 2**17, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            verbose = 0)
})
auc()



