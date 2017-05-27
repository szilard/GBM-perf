
## GBM Performance

Performance of various open source GBM implementations on the airline dataset (1M and 10M records).

GBM: `100` trees, depth `10`, learning rate `0.1`

On r4.8xlarge (32 cores, 250GB RAM)


Tool     |  Version        | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
---------|-----------------|-------------|---------------|-----------|------------
h2o      |  cran 3.10.4.6  |   25        |    140        |   0.762   |   0.776
xgboost  |  cran 0.6-4     |   21        |    290        |   0.750   |   0.751
lightgbm |  github 97ca38d |    6        |     50        |   0.764   |   0.775





