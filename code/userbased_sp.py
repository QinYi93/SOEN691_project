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

user_score = training.groupBy('userId').agg({"rating": "avg"})
user_score = user_score.withColumnRenamed("avg(rating)", "user-mean")
training = training.join(user_score, "userId")

training = training.withColumn("user-interaction", training.rating - (training['user-mean'] - global_mean))

als = ALS(maxIter=10, rank=20, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-interaction",
          coldStartStrategy="drop").setSeed(123)
model = als.fit(training)

predictions = model.transform(test)
predictions = predictions.join(user_score, "userId")
predictions = predictions.withColumn("final-rating", predictions.prediction + predictions['user-mean'] - global_mean)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="final-rating")
rmse = evaluator.evaluate(predictions)
print(str(rmse))

userRecs = model.recommendForAllUsers(10)
userRecs = userRecs.sort('userId', ascending=False).show(5)