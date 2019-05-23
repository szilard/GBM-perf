

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val d_train = spark.read.parquet("spark_ohe-train.parquet").cache()
val d_test = spark.read.parquet("spark_ohe-test.parquet").cache()
(d_train.count(), d_test.count())


val xgbParam = Map(
      "objective" -> "binary:logistic",
      "num_round" -> 100,
      "max_depth" -> 10,
      "eta" -> 0.1,
      "tree_method" -> "hist",
      "num_workers" -> 8,   // needs to be given manually (default =1)
      "missing" -> 0)

val gbm = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")


val pipeline = new Pipeline().setStages(Array(gbm))

val now = System.nanoTime
val model = pipeline.fit(d_train)
val elapsed = ( System.nanoTime - now )/1e9
elapsed


val predictions = model.transform(d_test)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("probability").setMetricName("areaUnderROC")
evaluator.evaluate(predictions)


