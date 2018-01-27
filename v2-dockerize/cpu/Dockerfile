FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y default-jdk-headless cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf

ENV MAKE="make -j4"

RUN install2.r ROCR data.table

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE=2018-01-27

RUN install2.r h2o

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .

RUN R -e 'devtools::install_github("Microsoft/LightGBM", subdir = "R-package")'


ARG CACHE_HACK=12

RUN git clone https://github.com/szilard/GBM-perf.git

CMD cd GBM-perf/v2-dockerize/cpu/run && \
    ln -s /train-1m.csv train.csv && \
    ln -s /test.csv test.csv && \
    echo "1m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R 2>/dev/null && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R

