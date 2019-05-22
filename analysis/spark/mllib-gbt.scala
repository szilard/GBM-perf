
// adapted from Joseph Bradley https://gist.github.com/jkbradley/1e3cc0b3116f2f615b3f


import org.apache.spark.ml.classification.GBTClassifier

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val d_train = spark.read.parquet("spark_ohe-train.parquet").cache()
val d_test = spark.read.parquet("spark_ohe-test.parquet").cache()
(d_train.count(), d_test.count())


val gbm = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").
  setMaxIter(100).setMaxDepth(10).setStepSize(0.1).
  setMaxBins(100).setMaxMemoryInMB(10240)     // max possible setMaxMemoryInMB (otherwise errors out)
val pipeline = new Pipeline().setStages(Array(gbm))


val now = System.nanoTime
val model = pipeline.fit(d_train)
val elapsed = ( System.nanoTime - now )/1e9
elapsed


val predictions = model.transform(d_test)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("probability").setMetricName("areaUnderROC")
evaluator.evaluate(predictions)


