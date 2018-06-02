FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties apt-transport-https

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    add-apt-repository 'deb [arch=amd64] https://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base


RUN apt-get install -y git wget libcurl4-openssl-dev default-jdk-headless cmake

RUN R CMD javareconf


ENV MAKE="make -j$(nproc)"

RUN R -e 'install.packages(c("ROCR","data.table","R6"), repos = c(CRAN = "https://cran.rstudio.com/"))'

RUN wget https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-1m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/train-10m.csv && \
    wget https://s3.amazonaws.com/benchm-ml--main/test.csv



ARG CACHE_DATE=2018-06-02

RUN R -e 'install.packages("h2o", repos = c(CRAN = "https://cran.rstudio.com/"))'

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make install -j


RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

#RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
#    cd LightGBM && \
#    mkdir build && cd build && cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
#    make -j$(nproc)

RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM/R-package && \
    sed -i 's/use_gpu <- FALSE/use_gpu <- TRUE/' src/install.libs.R && \
    sed -i 's/DUSE_GPU=ON/DUSE_GPU=1 -DOpenCL_LIBRARY=\/usr\/local\/cuda\/lib64\/libOpenCL.so -DOpenCL_INCLUDE_DIR=\/usr\/local\/cuda\/include\//' src/install.libs.R&& \
    R CMD INSTALL --build . --no-multiarch


ADD https://api.github.com/repos/szilard/GBM-perf/git/refs/heads/master version.json
## ^^^ hack to invalidate docker cache if repo gets updated
RUN git clone https://github.com/szilard/GBM-perf.git

RUN  apt-get clean && rm -rf /var/lib/apt/lists/*


CMD cd GBM-perf/gpu/run && \
    ln -s /test.csv test.csv && \
    ln -sf /train-0.1m.csv train.csv && \
    echo "0.1m:" && \
      # echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \
    ln -sf /train-1m.csv train.csv && \
    echo "1m:" && \
      # echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R && \      
    ln -sf /train-10m.csv train.csv && \
    echo "10m:" && \
      # echo -n "h2o " && R --slave < 1-h2o.R && \
      echo -n "xgboost " && R --slave < 2-xgboost.R && \
      echo -n "lightgbm " && R --slave < 3-lightgbm.R

