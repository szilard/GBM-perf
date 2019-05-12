FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y default-jdk-headless cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf

ENV MAKE="make -j4"

RUN install2.r ROCR data.table

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .

RUN R -e 'devtools::install_github("Laurae2/lgbdl"); lgbdl::lgb.dl()'


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git


ENV R_CMD="taskset -c 0 R --slave"

CMD cd GBM-perf/wip-testing/xgboost-old_laptop/run && \
    ln -s /test.csv test.csv && \
    ln -sf /train-10m.csv train.csv && \
    echo "10m:" && \
      echo -n "xgboost " && ${R_CMD} < 2-xgboost.R && \
      echo -n "lightgbm " && ${R_CMD} < 3-lightgbm.R

