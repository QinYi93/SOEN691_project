from pyspark.sql import Row, SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("collaborative_filtering").getOrCreate()
lines = spark.read.csv("../data/ratings.csv").rdd
ratingsRDD = lines.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)

(training, test) = ratings.randomSplit([0.7, 0.3], seed=123)

global_score = training.agg({"rating": "avg"}).collect()
global_mean = float(global_score[0][0])

user_score = training.groupBy('userId').agg({"rating": "avg"})
user_score = user_score.withColumnRenamed("avg(rating)", "user-mean")
training = training.join(user_score, "userId")

training = training.withColumn("user-interaction", training.rating - (training['user-mean'] - global_mean))

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=8, rank=16, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-interaction",
          coldStartStrategy="drop").setSeed(123)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
predictions = predictions.join(user_score, "userId")
predictions = predictions.withColumn("final-rating", predictions.prediction + predictions['user-mean'] - global_mean)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="final-rating")
rmse = evaluator.evaluate(predictions)
print(str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs = userRecs.sort('userId', ascending=False).show(10)