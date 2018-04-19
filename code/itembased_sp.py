from pyspark.sql import Row, SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("collaborative_filtering").getOrCreate()
lines = spark.read.csv("../data/ratings.csv").rdd
ratingsRDD = lines.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)

(training, test) = ratings.randomSplit([0.9, 0.1], seed=123)

global_score = training.agg({"rating": "avg"}).collect()
global_mean = float(global_score[0][0])

item_score = training.groupBy('movieId').agg({"rating": "avg"})
item_score = item_score.withColumnRenamed("avg(rating)", "item-mean")
training = training.join(item_score, "movieId")

training = training.withColumn("item-interaction",
                               training.rating - (training['item-mean'] - global_mean))

als = ALS(maxIter=10, rank=20, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="item-interaction",
          coldStartStrategy="drop").setSeed(123)
model = als.fit(training)

predictions = model.transform(test)
predictions = predictions.join(item_score, "movieId")
predictions = predictions.withColumn("final-rating", predictions.prediction + predictions['item-mean'] - global_mean)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="final-rating")
rmse = evaluator.evaluate(predictions)
print(str(rmse))

userRecs = model.recommendForAllUsers(10)
userRecs = userRecs.sort('userId', ascending=False).show(5)