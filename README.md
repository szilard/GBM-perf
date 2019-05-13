
# GBM Performance

Performance of the top/most widely used open source gradient boosting machines (GBM)/ boosted trees (GBDT)
implementations (h2o, xgboost, lightgbm, catboost) 
on the airline dataset (100K, 1M and 10M records) and with `100` trees, depth `10`, learning rate `0.1`.



## Popularity of GBM implementations

Poll conducted via twitter (April, 2019):

![](poll.png)



## How to run/reproduce the benchmark

Installing to latest software versions and running/timing is easy and fully automated with docker: 

### CPU

(requires docker)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/cpu
sudo docker build --build-arg CACHE_DATE=$(date +%Y-%m-%d) -t gbmperf_cpu .
sudo docker run --rm gbmperf_cpu
```

### GPU

(requires docker, NVIDIA drivers and the `nvidia-docker` utility)

```
git clone https://github.com/szilard/GBM-perf.git
cd GBM-perf/gpu
sudo docker build -t gbmperf_gpu .
sudo nvidia-docker run --rm gbmperf_gpu
```



## Results

### CPU 

r4.8xlarge (32 cores, but run on physical cores only/no hyperthreading) with software as of 2019-04-29:

Tool         | Time[s] 100K | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
-------------|--------------|-------------|---------------|-----------|------------
h2o          |   16         |   20        |    100        |   0.762   |   0.776
xgboost      |   3.8        |   12        |     78        |   0.749   |   0.755
**lightgbm**     |   **2.4**    |    **5.2**  |     42        |   0.764   |   0.774
catboost     |   5.4        |   50        |    490        |   0.740   |   0.744 


### GPU

p3.2xlarge (1 GPU, Tesla V100) with software as of 2019-04-29:

Tool            | Time[s] 100K | Time[s] 1M  |  Time[s] 10M  |   AUC 1M  |   AUC 10M
----------------|--------------|-------------|---------------|-----------|------------
h2o xgboost     |   9          |    14       |     60        |   0.749   |   0.756  
**xgboost**         | **2.4**      |  **4.8**    |   **13**      |   0.750   |   0.756
lightgbm        |   10         |    16       |     67        |   0.766   |   0.774
catboost        |   3.9        |    10       |    135        |   0.742   |   0.750 



## Additional results 

Some additional studies obtained "manually" (not fully automated with docker as the main benchmark above).
Thanks [@Laurae2](https://github.com/Laurae2) for lots of help with some if these. 

### Faster CPUs

AWS has now better CPUs than r4.8xlarge (Xeon E5-2686 v4 2.30GHz, 32 cores), for example with higher CPU frequency 
c5.9xlarge (Xeon Platinum 8124M 3.00GHz, 36 cores) or more number of cores 
m5.12xlarge (Xeon Platinum 8175M 2.50GHz, 48 cores).

c5 and m5 are typically 20-50% faster than r4, for larger data more cores (m5) is the best, 
for smaller data high-frequency CPU (c5) is the best. Nevertheless, **the ranking of libs by
training time stays the same** for a given data size when changing CPU. More details
[here](https://github.com/szilard/GBM-perf/issues/13).


### Multi-socket CPUs

Most high-end servers have nowadays more than 1 CPU on the motherboard. For example c5.18xlarge has 2 CPUs
(2x of the c5.9xlarge CPUs mentioned above), same for r4.16xlarge or m5.24xlarge. There are even EC2 instances with 
4 CPUs e.g. x1.32xlarge (128 cores) or more.

One would think more CPU cores means higher training speed, though because of RAM topology and NUMA, **most of the above tools
run slower on 2 CPUs than 1 CPU!!** (with the exception of h2o for large data). The slowdown might be pretty 
dramatic, e.g. 2x for lightgbm or 3-5x for xgboost for the larger data in this benchmark. If you don't know about this, 
you will pay more money for a larger instance and get actually much slower training. More details
[here](https://github.com/szilard/GBM-perf/issues/13) and 
[here](https://github.com/szilard/GBM-multicore).


### 100M records and RAM usage

Results on the fastest CPU (most cores, 1 socket, see above why this is the fastest) and the fastest GPU on EC2.
The data is obtained by replicating the 10M dataset 10x, so the AUC is not indicative of a learning curve, just used to
see if it is equal approximately the 10M AUC (it should be).

For the CPU runs, "RAM train" is measured as the increase in memory usage during training (on top of the RAM used by the data). 
For the GPU runs, the "GPU memory" usage is the total GPU memory used (cannot separate training from copies of the data),
while the "extra RAM" is the additional RAM used by some of the tools (on the CPU) if any.

CPU (m5.12xlarge):

Tool              | time [s]   | AUC       | RAM train [GB]
------------------|------------|-----------|-------------------------
h2o               | 520        |  0.775    |   8
xgboost           | 510        |  0.751    |  15
**lightgbm**      | **310**    |  0.774    |   **5**
catboost          | 3360       |  0.723 ?! |  140

GPU (Tesla V100):

Tool              | time [s]    |  AUC      | GPU mem [GB]   | extra RAM [GB]
------------------|-------------|-----------|----------------|----------------
h2o xgboost       | 270         | 0.755     | 4              | 30
**xgboost**       | **80**      | 0.756     | 6              | **0**
lightgbm          | 400         | 0.774     | 3              | 6
catboost          | crash (OOM) |           | >16            | 14

Note that catboost CPU achieves lower AUC vs the 10M dataset (might be due to the way of binning or some other approximation).
catboost GPU crashes out-of-memory on the 16GB GPU (while this doesn't tell us how fast it would run with more GPU RAM, 
the results with 10M data indicate that it would be slow compared to the other libs).

h2o xgboost on GPU is slower than native xgboost on GPU and also adds
a lot of overhead in RAM usage ("extra RAM") (this must be due to some pre- and post-processing of data in h2o as one can
see by looking at the GPU utilization patterns as discussed next).

More details [here](https://github.com/szilard/GBM-perf/issues/14).


### GPU utilization patterns

For the GPU runs, it is interesting to observe the GPU utilization patterns and also the CPU utilization meanwhile
(usually 1 CPU thread).

xgboost uses GPU at ~80% and 1 CPU core at 100%.

h2o xgboost shows 3 phases: first only using CPU at ~30% (all cores) and no GPU, then GPU at ~70% and CPU at 100%, then
no GPU and CPU at 100%. This means 3-4x longer training time vs native xgboost. 

lightgbm uses GPU at 5-10% and meanwhile CPU at 100% (all cores). It can be made to use 1 CPU core only (`nthread = 1`), but
then it may be slower.

catboost uses GPU at ~80% and 1 CPU core at 100%. Unlike the other tools catboost takes all the GPU memory available when it
starts training no matter of the data size (so we don't know how much memory it needs by using the standard monitoring tools).

More details [here](https://github.com/szilard/GBM-perf/issues/11).


### Spark MLlib 

In my previous broader benchmark of ML libraries, Spark MLlib GBT (and random forest as well) performed very poorly 
(10-100x running time vs top libs, 10-100x memory usage and an accuracy issue for larger data) and therefore it
was not included in the current GBM/GBT benchmark. However, people might still be interested if there has been any
improvements since 2016 and Spark 2.0.

With Spark 2.4.2 as of 2019-05-05  the accuracy issue for larger data has been fixed, but the
speed and the memory footprint did not improve:

size  | time lgbm [s] | time spark [s] | ratio | AUC lgbm | AUC spark
------|---------------|----------------|-------|----------|-------------
100K  |           2.4 |           1020 | 425   |    0.730 | 0.721
1M    |           5.2 |           1380 | 265   |    0.764 | 0.748
10M   |            42 |           8390 | 200   |    0.774 | 0.755

(compared to lighgbm CPU) (Spark code [here](https://github.com/szilard/GBM-perf/tree/master/wip-testing/spark))

So Spark MLlib GBT is still 100x slower than the top tools. In case you are wondering if more nodes or
bigger data would help, the answer in nope (see below).

#### Spark MLlib on 100M records and RAM usage

Besides being slow, Spark also uses 100x RAM compared to the top tools. In fact, on 100M records 
(20GB after being loaded from disk and cached in RAM) it crashes out-of-memory even on servers with almost 1 TB RAM.

      |       | 100M      |       |            | 10M      |       |  
----- | ----- | --------- | ----- | ---------- | -------- | ----- | --
trees | depth | time [s]  | AUC   | RAM [GB]   | time [s] | AUC   | RAM [GB]
1     | 1     | 1150      | 0.634 | 620        | 70       | 0.635 | 110
1     | 10    | 1350      | 0.712 | 620        | 90       | 0.712 | 112
10    | 10    | 7850      | 0.731 | 780        | 830      | 0.731 | 125
100   | 10    | crash OOM |       | >960 (OOM) | 8390     | 0.755 | 230

(100M ran on x1e.8xlarge [32 cores, 960GB RAM], 10M ran on r4.8xlarge [32 cores, 240GB RAM])

(compare this with 100M records 100 trees depth 10, lightgbm 5GB RAM usage)

More details [here](https://github.com/szilard/GBM-perf/issues/18). 

Note the situation is much better for linear models in Spark MLlib, only 3-4x slower and 10x more memory
footprint vs h2o for example, see results [here](https://github.com/szilard/GBM-perf/issues/20) (and training
linear models is much much faster than trees, so training times are reasonable even for large data).


## Recommendations

If you **don't have a GPU, lightgbm** (CPU) trains the fastest.

If you **have a GPU, xgboost** (GPU) is also very fast (and depending on the data, your hardware etc.
often faster than the above mentioned lightgbm on CPU).

If you consider deployment, **h2o has the best ways to deploy** as a real-time
(fast scoring) application.

Note, however, there are a lot more other criteria to consider when you choose which tool
to use.

More info in my eRum 2018 R conference talk 
(video recording [here](https://www.youtube.com/watch?v=DqS6EKjqBbY),
slides [here](https://speakerdeck.com/szilard/better-than-deep-learning-gradient-boosting-machines-gbm-in-r-erum-conference-budapest-may-2018)), 
and a summary comparison table here:

![](comparison_table.png)


