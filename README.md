
## GBM Performance

Performance of various open source GBM implementations on the airline dataset (1M and 10M records).

GBM: `100` trees, depth `10`, learning rate `0.1`


On r4.8xlarge (32 cores, 250GB RAM)

Tool         |  Version        | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
-------------|-----------------|-------------|---------------|-----------|------------
h2o          |  cran 3.10.4.6  |   25        |    140        |   0.762   |   0.776
xgboost      |  cran 0.6-4     |   20        |    290        |   0.750   |   0.751
xgboost hist |  github 6776292 |   20        |    170        |   0.766   |   0.772
lightgbm     |  github 97ca38d |    6        |     50        |   0.764   |   0.775


With GPU support on p2.xlarge (Tesla K80, 12GB)

Tool            |  Version               | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|------------------------|-------------|---------------|-----------|------------
h2o xgboost     |  deep water 3.11.0.266 |   20        |    180        |   0.715   |   0.708
xgboost hist    |  github 64c8f6f        |   6         |    50         |   0.750   |   0.740
lightgbm        |  github 1d5867b        |   30        |    120        |   0.771   |   0.789

on g3.4xlarge (Tesla M60, 8GB)

Tool            |  Version               | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|------------------------|-------------|---------------|-----------|------------
lightgbm        |  github 1d5867b        |   20        |    50         |   0.771   |   0.789


----------------------------------------------

### v2: dockerizing (WIP)

Automated install to latest h2o/xgboost/lightgbm versions and automated running/timing. 

So far done for CPU versions:

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/v2-dockerize/cpu
sudo docker build -t gbmperf_cpu .
sudo docker run gbmperf_cpu
```

Results on r4.8xlarge (32 cores) with software as of 2018-01-27:

```
1m:
h2o 22.762 0.7623672
xgboost 33.029 0.7507367
lightgbm 6.006 0.7660324

10m:
h2o 91.364 0.7763126
xgboost 499.969 0.7519515
lightgbm 46.375 0.7739303
```







