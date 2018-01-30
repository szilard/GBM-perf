
## GBM Performance

Performance of various open source GBM implementations (h2o, xgboost, lightgbm) on the airline dataset (1M and 10M records).

GBM: `100` trees, depth `10`, learning rate `0.1`



### Run

Installing to latest software versions and running/timing is fully automated with docker: 

#### CPU

(requires docker)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/v2-dockerize/cpu
sudo docker build -t gbmperf_cpu .
sudo docker run --rm gbmperf_cpu
```

#### GPU

(requires docker, nvidia drivers and the `nvidia-docker` utility)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/v2-dockerize/gpu
sudo docker build -t gbmperf_gpu .
sudo nvidia-docker run --rm gbmperf_gpu
```



### Results


#### CPU 

r4.8xlarge (32 cores) with software as of 2018-01-27:

Tool         | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
-------------|-------------|---------------|-----------|------------
h2o          |   21        |     90        |   0.762   |   0.776
xgboost      |   16        |    150        |   0.749   |   0.755
lightgbm     |    6        |     47        |   0.766   |   0.774


#### GPU

p3.2xlarge (1 GPU, Tesla V100) with software as of 2018-01-29:

Tool            | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|-------------|---------------|-----------|------------
h2o xgboost     |   TODO      |    TODO       |   TODO    |   TODO
xgboost         |   8         |    25         |   0.748   |   0.756
lightgbm        |   20        |    75         |   0.766   |   0.774



----------------------------------------

Old results (until h2o GPU above is done) on p2.xlarge:

Tool            |  Version               | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|------------------------|-------------|---------------|-----------|------------
h2o xgboost     |  deep water 3.11.0.266 |   20        |    180        |   0.715   |   0.708



