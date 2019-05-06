#!/bin/bash

SPARK_ROOT=spark-2.4.2-bin-hadoop2.7

TOOL=$1   ## mllib-gbt 
SIZE=$2   ## 0.1m 1m 10m

ln -sf spark_ohe-train-${SIZE}.parquet spark_ohe-train.parquet
ln -sf spark_ohe-test-${SIZE}.parquet spark_ohe-test.parquet
${SPARK_ROOT}/bin/spark-shell --master local[*] --driver-memory 100G --executor-memory 100G < $TOOL.scala
rm -r spark_ohe-train.parquet spark_ohe-test.parquet

