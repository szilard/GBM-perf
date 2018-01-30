
## GBM Performance

Performance of various open source GBM implementations (h2o, xgboost, lightgbm) on the airline dataset (1M and 10M records).

GBM: `100` trees, depth `10`, learning rate `0.1`



### Run

Install to latest software versions and run timing fully automated with docker: 

#### CPU

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/v2-dockerize/cpu
sudo docker build -t gbmperf_cpu .
sudo docker run --rm gbmperf_cpu
```

#### GPU

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/v2-dockerize/gpu
sudo docker build -t gbmperf_gpu .
sudo nvidia-docker run --rm gbmperf_gpu
```



### Results


#### CPU 

r4.8xlarge (32 cores) with software as of 2018-01-27:

```
Tool / time (s) / AUC

1m:
h2o 21.415 0.7623672
xgboost 16.096 0.7494959
lightgbm 6.117 0.7660324

10m:
h2o 89.497 0.7763126
xgboost 151.108 0.7551197
lightgbm 47.291 0.7739303
```

Tool         | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
-------------|-------------|---------------|-----------|------------
h2o          |   21        |     90        |   0.762   |   0.776
xgboost      |   16        |    150        |   0.749   |   0.755
lightgbm     |    6        |     47        |   0.766   |   0.774


#### GPU



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



