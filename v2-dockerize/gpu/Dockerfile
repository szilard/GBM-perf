FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties apt-transport-https

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    add-apt-repository 'deb [arch=amd64] https://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base


RUN apt-get install -y git wget libcurl4-openssl-dev default-jdk-headless cmake

RUN  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf


ENV MAKE="make -j4"

RUN R -e 'install.packages(c("ROCR","data.table"), repos = c(CRAN = "https://cran.rstudio.com/"))'

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE=2018-01-27

RUN R -e 'install.packages("h2o", repos = c(CRAN = "https://cran.rstudio.com/"))'

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make install -j




ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git

CMD cd GBM-perf/v2-dockerize/gpu/run && \
    ln -s /train-1m.csv train.csv && \
    ln -s /test.csv test.csv && \
    echo "1m:" && \
  #    echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R
    ## && \
  #    echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
 #   ln -sf /train-10m.csv train.csv && \
 #   echo "10m:" && \
 #     echo -n "h2o " && R --slave < 1-h2o.R && \
 #     echo -n "xgboost " && R --slave < 2-xgboost.R && \
 #     echo -n "lightgbm " && R --slave < 3-lightgbm.R

