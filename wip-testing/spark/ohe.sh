#!/bin/bash

SPARK_ROOT=spark-2.4.2-bin-hadoop2.7

for SIZE in 0.1m 1m 10m; do
   ln -sf train-${SIZE}.csv train.csv
   ${SPARK_ROOT}/bin/spark-shell --master local[*] --driver-memory 100G --executor-memory 100G < ohe.scala
   mv spark_ohe-train.parquet spark_ohe-train-${SIZE}.parquet
   mv spark_ohe-test.parquet spark_ohe-test-${SIZE}.parquet
done
rm train.csv

