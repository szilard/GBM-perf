
## GBM Performance

Performance of various open source GBM implementations (h2o, xgboost, lightgbm) on the airline dataset (1M and 10M records).

GBM: `100` trees, depth `10`, learning rate `0.1`



### Run

Installing to latest software versions and running/timing is fully automated with docker: 

#### CPU

(requires docker)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/cpu
sudo docker build -t gbmperf_cpu .
sudo docker run --rm gbmperf_cpu
```

#### GPU

(requires docker, nvidia drivers and the `nvidia-docker` utility)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/gpu
sudo docker build -t gbmperf_gpu .
sudo nvidia-docker run --rm gbmperf_gpu
```



### Results


#### CPU 

r4.8xlarge (32 cores) with software as of 2018-06-02:

Tool         | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
-------------|-------------|---------------|-----------|------------
h2o          |   21        |     90        |   0.762   |   0.776
xgboost      |   16        |    130        |   0.749   |   0.755
lightgbm     |    **6**    |     47        |   0.766   |   0.774


#### GPU

p3.2xlarge (1 GPU, Tesla V100) with software as of 2018-01-29:

Tool            | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|-------------|---------------|-----------|------------
h2o xgboost     |   18        |   error       |   0.712   |  error
xgboost         |   8         |   **25**      |   0.748   |   0.756
lightgbm        |   20        |    75         |   0.766   |   0.774



### Recommendations

If you don't have a GPU, lightgbm (CPU) trains the fastest.

If you have a GPU, xgboost (GPU) is also very fast (and depending on the data, 
often the fastest).

If you consider deployment, h2o has the best ways to deploy as a real-time
(fast scoring) application.

Note, however, there are a lot more other criteria to consider when you choose which tool
to use.

More info in my eRum 2018 R conference talk 
(video recording [here](https://www.youtube.com/watch?v=DqS6EKjqBbY),
slides [here](https://speakerdeck.com/szilard/better-than-deep-learning-gradient-boosting-machines-gbm-in-r-erum-conference-budapest-may-2018)), and a summary comparison table here:

![](comparison_table.png)



