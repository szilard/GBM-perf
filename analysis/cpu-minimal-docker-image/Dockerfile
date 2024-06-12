FROM r-base

RUN apt-get update && \
    apt-get install -y cmake git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV MAKE="make -j4"

RUN install2.r ROCR data.table jsonlite markdown knitr R6

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL . && \
    cd / && rm -rf xgboost

RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && Rscript build_r.R && \
    cd / && rm -rf LightGBM


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git


ENV R_CMD="R --slave"

CMD cd GBM-perf/cpu/run && \
    ln -s /test.csv test.csv && \
    ln -sf /train-1m.csv train.csv && \
    echo "1m:" && \
      echo -n "xgboost " && ${R_CMD} < 2-xgboost.R && \
      echo -n "lightgbm " && ${R_CMD} < 3-lightgbm.R
