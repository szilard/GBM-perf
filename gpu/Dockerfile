FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common apt-transport-https

RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    apt-get update && \
    apt-get install -y r-base

RUN apt-get install -y git wget libcurl4-openssl-dev default-jdk-headless libssl-dev libxml2-dev cmake libgit2-dev \
                       libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev \
                       libtiff5-dev libjpeg-dev

RUN R CMD javareconf

ARG MAKE

RUN R -e 'install.packages(c("ROCR","data.table","R6","devtools","RCurl","jsonlite"), repos = "https://cran.rstudio.com/")'

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv


ARG CACHE_DATE

RUN R -e 'install.packages("h2o", repos = "http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")'


# xgboost compilation errors out, so use CUDA-built binaries (supplied in the release section on github for the latest release)

#RUN git clone --recursive https://github.com/dmlc/xgboost && \
#    cd xgboost && \
#    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
#    make install -j

# release 2.0.3 (2023-12-19)
RUN wget https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_2.0.0/xgboost_r_gpu_linux_82d846bbeb83c652a0b1dff0e3519e67569c4a3d.tar.gz
RUN R CMD INSTALL xgboost_r_gpu_linux_82d846bbeb83c652a0b1dff0e3519e67569c4a3d.tar.gz


# lightgbm compilation errors out

#RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev opencl-headers clinfo
#RUN mkdir -p /etc/OpenCL/vendors && \
#    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd   ## otherwise lightgm segfaults at runtime (compiles fine without it)

#RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM && \
#    Rscript build_r.R --use-gpu --opencl-library=/usr/lib/x86_64-linux-gnu/libOpenCL.so --boost-librarydir=/usr/lib/x86_64-linux-gnu


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
     # echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R && \
    echo "0.1m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
     # echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R && \
    ln -sf /train-1m.csv train.csv && \
    echo "1m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
     # echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R && \
    ln -sf /train-10m.csv train.csv && \
    echo "10m:" && \
      echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
     # echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
      echo -n "catboost " && R --slave < 4-catboost.R

