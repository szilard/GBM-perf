suppressMessages({
library(data.table)
library(ROCR)
library(gbm)
library(Matrix)
})

set.seed(123)

d_train <- fread("train.csv", showProgress=FALSE, stringsAsFactors=TRUE)
d_test <- fread("test.csv", showProgress=FALSE, stringsAsFactors=TRUE)

d_train$dep_delayed_15min <- ifelse(d_train$dep_delayed_15min=="Y",1,0)
d_test$dep_delayed_15min <- ifelse(d_test$dep_delayed_15min=="Y",1,0)


cat(system.time({
  md <- gbm(dep_delayed_15min ~ ., data = d_train, distribution = "bernoulli", 
            n.trees = 100, interaction.depth = 10, shrinkage = 0.1)
})[[3]]," ",sep="")


phat <- predict(md, newdata = d_test, n.trees = 100, type = "response")
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
cat(performance(rocr_pred, "auc")@y.values[[1]],"\n")


