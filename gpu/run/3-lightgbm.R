suppressMessages({
library(data.table)
library(ROCR)
library(lightgbm)
library(Matrix)
})

set.seed(123)

d_train <- fread("train.csv", showProgress=FALSE)
d_test <- fread("test.csv", showProgress=FALSE)

d_all <- rbind(d_train, d_test)
d_all$dep_delayed_15min <- ifelse(d_all$dep_delayed_15min=="Y",1,0)

d_all_wrules <- lgb.convert_with_rules(d_all)       
d_all <- d_all_wrules$data
cols_cats <- names(d_all_wrules$rules) 

d_train <- d_all[1:nrow(d_train)]
d_test <- d_all[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test))]

p <- ncol(d_all)-1
dlgb_train <- lgb.Dataset(data = as.matrix(d_train[,1:p]), label = d_train$dep_delayed_15min, free_raw_data = FALSE)

cat(system.time({
  md <- lgb.train(data = dlgb_train, 
            objective = "binary", 
            nrounds = 100, num_leaves = 512, learning_rate = 0.1, 
            categorical_feature = cols_cats,
            device = "gpu",
            verbose = 0)
})[[3]]," ",sep="")


phat <- predict(md, data = as.matrix(d_test[,1:p]))
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
cat(performance(rocr_pred, "auc")@y.values[[1]],"\n")


