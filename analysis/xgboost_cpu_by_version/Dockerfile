FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV MAKE="make -j4"

RUN install2.r ROCR data.table

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

ARG VER

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git checkout tags/$VER && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git


ARG CORES_1SO_NOHT

CMD cd GBM-perf/analysis/xgboost_cpu_by_version && \
    ln -s /test.csv test.csv && \
    ln -sf /train-1m.csv train.csv && \
    echo -n "xgboost 1 core" && taskset -c 0 R --slave < 2-xgboost.R && \
    echo -n "xgboost 1 socket & no HT" && taskset -c ${CORES_1SO_NOHT} R --slave < 2-xgboost.R && \
    echo -n "xgboost all cores" && R --slave < 2-xgboost.R 

