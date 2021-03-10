
library(h2o)
h2o.init(max_mem_size = "10g", nthreads = -1)


dx_train <- h2o.importFile("train-1m.csv")
dx_valid <- h2o.importFile("valid.csv")
dx_test <- h2o.importFile("test.csv")


## to have same normalization as for the other DL libs that don't auto normalize 
dx_train$DepTime <- dx_train$DepTime/2500
dx_valid$DepTime <- dx_valid$DepTime/2500
dx_test$DepTime <- dx_test$DepTime/2500

dx_train$Distance <- log10(dx_train$Distance)/4
dx_valid$Distance <- log10(dx_valid$Distance)/4
dx_test$Distance <- log10(dx_test$Distance)/4


Xnames <- names(dx_train)[which(names(dx_train)!="dep_delayed_15min")]



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            ## DEFAULT: activation = "Rectifier", hidden = c(200,200), 
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

## print epochs:  1: best AUC (on validation)  2: early stopping
d_scoring <- md@model$scoring_history
d_scoring[d_scoring$validation_auc==max(d_scoring$validation_auc, na.rm=TRUE),]

#   user  system elapsed
#  2.097   0.032 232.004
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7305076



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), input_dropout_ratio = 0.2,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.895   0.009  81.844
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7288232



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(50,50,50,50), 
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.767   0.012  69.534
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7328179



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20,20),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.625   0.004  49.193
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7319457



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(20),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.576   0.007  40.566
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7266034



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(10),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.602   0.012  43.195
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7307281



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(5),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.519   0.005  37.024
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7282662



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(1),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  0.497   0.005  31.882
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7105887



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), l1 = 1e-5, l2 = 1e-5, 
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  2.145   0.039 231.927
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7311061



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "RectifierWithDropout", hidden = c(200,200,200,200), hidden_dropout_ratios=c(0.2,0.1,0.1,0),
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  3.632   0.084 437.021
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7287979




system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.95, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  1.845   0.018 209.470
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7188023    



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  2.309   0.020 266.621
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7277615    



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.9999, epsilon = 1e-08,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  2.188   0.011 252.242
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7253196



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-06,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  2.766   0.024 330.894
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7170915



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            rho = 0.999, epsilon = 1e-09,  ## default:  rho = 0.99, epsilon = 1e-08
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  2.250   0.020 259.506
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7279227



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, ## default: rate = 0.005, rate_decay = 1, momentum_stable = 0,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  3.504   0.040 419.666
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7276413



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.001, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  5.501   0.076 662.537
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7328587



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  3.377   0.032 403.220
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7234964



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  4.503   0.024 534.935
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7340129



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-04, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.99,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#    user   system  elapsed
#  18.461    0.222 2247.658
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7302286



system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e5, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  4.596   0.081 535.119
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7339773


system.time({
  md <- h2o.deeplearning(x = Xnames, y = "dep_delayed_15min", training_frame = dx_train, validation_frame = dx_valid,
            activation = "Rectifier", hidden = c(200,200), 
            adaptive_rate = FALSE, rate = 0.01, rate_annealing = 1e-05, 
            momentum_start = 0.5, momentum_ramp = 1e4, momentum_stable = 0.9,
            epochs = 100, stopping_rounds = 3, stopping_metric = "AUC", stopping_tolerance = 0) 
})
h2o.performance(md, dx_test)@metrics$AUC

#   user  system elapsed
#  4.319   0.048 504.486
#> h2o.performance(md, dx_test)@metrics$AUC
#[1] 0.7336099

