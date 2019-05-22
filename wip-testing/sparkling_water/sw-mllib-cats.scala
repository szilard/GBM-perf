
import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(spark) 
import h2oContext._ 


import org.apache.spark.ml.h2o.algos.H2OGBM

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val d_train0 = spark.read.parquet("spark_ohe-train.parquet").cache()
val d_test0 = spark.read.parquet("spark_ohe-test.parquet").cache()
(d_train0.count(), d_test0.count())


val d_train = d_train0.select("Month","DayofMonth","DayOfWeek","DepTime","UniqueCarrier","Origin","Dest","Distance","dep_delayed_15min").cache()
val d_test = d_test0.select("Month","DayofMonth","DayOfWeek","DepTime","UniqueCarrier","Origin","Dest","Distance","dep_delayed_15min").cache()
(d_train.count(), d_test.count())


val gbm = new H2OGBM().setLabelCol("dep_delayed_15min").
  setNtrees(100).setMaxDepth(10).setLearnRate(0.1)          // .setMaxBins(100)   not implemented??
val pipeline = new Pipeline().setStages(Array(gbm))


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

