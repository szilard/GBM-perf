library(data.table)
library(ranger)
library(ROCR)

d_train <- fread("train-1m.csv")
d_test <- fread("test.csv")

d_train$dep_delayed_15min <- as.factor(d_train$dep_delayed_15min)
d_test$dep_delayed_15min  <- as.factor(d_test$dep_delayed_15min)

auc <- function() {
  phat <- predictions(predict(md, data = d_test))[,"Y"]
  rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
  performance(rocr_pred, "auc")@y.values[[1]]
}


system.time({
  md <- ranger(dep_delayed_15min ~ ., d_train, 
          num.trees = 100, max.depth = 10, probability = TRUE, write.forest = TRUE)
})
auc()



system.time({
  md <- ranger(dep_delayed_15min ~ ., d_train, 
          num.trees = 100, max.depth = 15, probability = TRUE, write.forest = TRUE)
})
auc()


system.time({
  md <- ranger(dep_delayed_15min ~ ., d_train, 
          num.trees = 100, max.depth = 20, probability = TRUE, write.forest = TRUE)
})
auc()


