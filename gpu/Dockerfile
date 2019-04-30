FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties apt-transport-https

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    add-apt-repository 'deb [arch=amd64] https://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base


RUN apt-get install -y git wget libcurl4-openssl-dev default-jdk-headless libssl-dev

## xgboost: CMake 3.12 or higher is required
RUN apt-get install -y python-pip && pip install cmake --upgrade

RUN R CMD javareconf

ENV MAKE="make -j$(nproc)"

RUN R -e 'install.packages(c("ROCR","data.table","R6","devtools","RCurl","jsonlite"), repos = "https://cran.rstudio.com/")'

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

RUN R -e 'install.packages("h2o", repos = "http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")'

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make install -j

RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev opencl-headers clinfo
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd   ## otherwise lightgm segfaults at runtime (compiles fine without it)
RUN R -e 'devtools::install_github("Laurae2/lgbdl"); lgbdl::lgb.dl(use_gpu = TRUE)'

RUN R -e 'devtools::install_github("catboost/catboost", subdir = "catboost/R-package")'


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git

RUN  apt-get clean && rm -rf /var/lib/apt/lists/*


CMD cd GBM-perf/gpu/run && \
    ln -s /test.csv test.csv && \
    ln -sf /train-0.1m.csv train.csv && \
    echo "warmup:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R && \
    echo "0.1m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R && \
    ln -sf /train-1m.csv train.csv && \
    echo "1m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \      
      echo -n "catboost " && R --slave < 4-catboost.R && \
    ln -sf /train-10m.csv train.csv && \
    echo "10m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R

