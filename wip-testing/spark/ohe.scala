
// from Joseph Bradley https://gist.github.com/jkbradley/1e3cc0b3116f2f615b3f
// modifications by Xusen Yin https://github.com/szilard/benchm-ml/commit/db65cf000c9b1565b6e93d2d10c92dd646644d85
// further changes by @szilard 


import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.DoubleType

val loader = spark.read.format("com.databricks.spark.csv").option("header", "true")
val trainDF = loader.load("train.csv")
val testDF = loader.load("test.csv")

val fullDF0 = trainDF.withColumn("isTrain", lit(true)).unionAll(testDF.withColumn("isTrain", lit(false)))

val fullDF = fullDF0.withColumn("DepTime", col("DepTime").cast(DoubleType)).withColumn("Distance", col("Distance").cast(DoubleType))

fullDF.printSchema
fullDF.show(5)


val res = new RFormula().setFormula("dep_delayed_15min ~ . - isTrain").fit(fullDF).transform(fullDF)

res.printSchema
res.show(5)

val finalTrainDF = res.where(col("isTrain"))
val finalTestDF = res.where(!col("isTrain"))

finalTrainDF.write.mode("overwrite").parquet("spark_ohe-train.parquet")
finalTestDF.write.mode("overwrite").parquet("spark_ohe-test.parquet")


