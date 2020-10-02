library(data.table)
library(randomForestSRC)
library(ROCR)

d_train <- fread("train-1m.csv", stringsAsFactors=TRUE)
d_test <- fread("test.csv", stringsAsFactors=TRUE)

auc <- function() {
  phat <- predict(md, data = d_test)$predicted[,"Y"]
  rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
  performance(rocr_pred, "auc")@y.values[[1]]
}
## not working (yet):
## Number of predictions in each run must be equal to the number of labels for each run.
## length(phat)=1000000 ??


system.time({
  md <- rfsrc(dep_delayed_15min ~ ., d_train, 
          ntree = 100, nodedepth = 10)
})
auc()

#    user   system  elapsed
#1180.906    5.416   94.359

