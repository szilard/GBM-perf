FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y default-jdk-headless cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf

ENV MAKE="make -j16"

RUN install2.r ROCR data.table markdown

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

#RUN install2.r RCurl && \
#    install2.r -r http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R h2o

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .

RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && Rscript build_r.R

#RUN R -e 'devtools::install_github("catboost/catboost", subdir = "catboost/R-package")'


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git


CMD cd GBM-perf/analysis/multicore-c7metal/run && \
    ln -s /test.csv test.csv && \
    bash run.sh
