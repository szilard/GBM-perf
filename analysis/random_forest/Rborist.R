library(data.table)
library(ROCR)
library(Matrix)
library(Rborist)

set.seed(123)

d_train <- fread("train-1m.csv")
d_test <- fread("test.csv")

X_train_test <- sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
X_train <- X_train_test[1:nrow(d_train),]
X_test <- X_train_test[(nrow(d_train)+1):(nrow(d_train)+nrow(d_test)),]

auc <- function() {
  phat <- predict(md, newdata = X_test, ctgCensus="prob")$prob[,"Y"]
  rocr_pred <- prediction(phat, d_test$dep_delayed_15min == "Y")
  performance(rocr_pred, "auc")@y.values[[1]]
}


system.time({
    md <- Rborist(X_train, as.factor(d_train$dep_delayed_15min), nLevel=10, nTree=100, predProb = 1/sqrt(length(X_train@x)/nrow(X_train)), thinLeaves=TRUE)
})
auc()


system.time({
    md <- Rborist(X_train, as.factor(d_train$dep_delayed_15min), nLevel=15, nTree=100, predProb = 1/sqrt(length(X_train@x)/nrow(X_train)), thinLeaves=TRUE)
})
auc()


system.time({
    md <- Rborist(X_train, as.factor(d_train$dep_delayed_15min), nLevel=20, nTree=100, predProb = 1/sqrt(length(X_train@x)/nrow(X_train)), thinLeaves=TRUE)
})
auc()


