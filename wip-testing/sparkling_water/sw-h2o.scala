
import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(spark) 
import h2oContext._ 


import _root_.hex.tree.gbm.GBM
import _root_.hex.tree.gbm.GBMModel.GBMParameters
import water.support.H2OFrameSupport

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val d_train = spark.read.parquet("spark_ohe-train.parquet").cache()
val d_test = spark.read.parquet("spark_ohe-test.parquet").cache()
(d_train.count(), d_test.count())


val now = System.nanoTime
val dx_train = asH2OFrame(d_train.select("Month","DayofMonth","DayOfWeek","DepTime","UniqueCarrier",
      "Origin","Dest","Distance","dep_delayed_15min"))
H2OFrameSupport.allStringVecToCategorical(dx_train)     
val elapsed = ( System.nanoTime - now )/1e9
elapsed

val dx_test = asH2OFrame(d_test.select("Month","DayofMonth","DayOfWeek","DepTime","UniqueCarrier",
      "Origin","Dest","Distance"))


val gbmParams = new GBMParameters()
    gbmParams._train = dx_train._key
    gbmParams._response_column = "dep_delayed_15min"
    gbmParams._ntrees = 100
    gbmParams._max_depth = 10
    gbmParams._learn_rate = 0.1
    gbmParams._nbins = 100
val gbm = new GBM(gbmParams)


val now = System.nanoTime
val gbm_md = gbm.trainModel.get
val elapsed = ( System.nanoTime - now )/1e9
elapsed


val predx = gbm_md.score(dx_test)
val pred0 = asDataFrame(predx)
val pred0ID = pred0.withColumn("id", monotonically_increasing_id())
val d_testID = d_test.withColumn("id", monotonically_increasing_id())
val predictions = pred0ID.join(d_testID, "id")

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("Y").setMetricName("areaUnderROC")
evaluator.evaluate(predictions)


