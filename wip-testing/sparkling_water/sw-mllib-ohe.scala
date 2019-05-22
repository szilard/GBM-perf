
import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(spark) 
import h2oContext._ 


import org.apache.spark.ml.h2o.algos.H2OGBM

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val d_train = spark.read.parquet("spark_ohe-train.parquet").cache()
val d_test = spark.read.parquet("spark_ohe-test.parquet").cache()
(d_train.count(), d_test.count())


val gbm = new H2OGBM().setLabelCol("label").setFeaturesCol("features").
  setNtrees(10).setMaxDepth(10).setLearnRate(0.1)    //.setMaxBins(100)   not implemented??
val pipeline = new Pipeline().setStages(Array(gbm))

// slow with OHE 10 trees 136 sec vs 100 trees 28 sec (m5.2xlarge 8 cores) -- 50x

val now = System.nanoTime
val model = pipeline.fit(d_train)
val elapsed = ( System.nanoTime - now )/1e9
elapsed


val predictions = model.transform(d_test)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction_output").setMetricName("areaUnderROC")
evaluator.evaluate(predictions)

// TODO:
//evaluator.evaluate(predictions)
//java.lang.IllegalArgumentException: requirement failed: Column prediction_output must be of type equal to one of the following types: [double, struct<type:tinyint,size:int,indices:array<int>,values:array<double>>] but was actually of type struct<value:double>.

