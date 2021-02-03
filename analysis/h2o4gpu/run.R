library(data.table)
library(ROCR)
library(h2o4gpu)

set.seed(123)

d_train <- fread("train.csv")
d_test <- fread("test.csv")


X_train_test <- Matrix::sparse.model.matrix(dep_delayed_15min ~ .-1, data = rbind(d_train, d_test))
n1 <- nrow(d_train)
n2 <- nrow(d_test)
X_train <- X_train_test[1:n1,]
X_test <- X_train_test[(n1+1):(n1+n2),]

y_train <- ifelse(d_train$dep_delayed_15min=='Y',1,0)
y_test <- ifelse(d_test$dep_delayed_15min=='Y',1,0)


system.time({
md <- h2o4gpu.gradient_boosting_classifier(n_estimators = 100L, max_depth = 10L, learning_rate = 0.1) %>% 
         fit(X_train, y_train)
})


phat <- predict(md, X_test, type = "prob")[,"1"]
rocr_pred <- prediction(phat, d_test$dep_delayed_15min)
performance(rocr_pred, "auc")@y.values[[1]]


