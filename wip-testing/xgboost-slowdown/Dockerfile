FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV MAKE="make -j4"

RUN install2.r ROCR data.table

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv

ARG VER=master   
## default / pass other VER in `docker build --build-arg VER=v0.80`

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git checkout $VER && \
    git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json  
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git

CMD cd GBM-perf/wip-testing/xgboost-slowdown && \
    ln -s /test.csv test.csv && \
    ln -sf /train-1m.csv train.csv && \
    echo "1m:" && \
      echo -n "xgboost " && R --slave < 2-xgboost.R 

