from pyspark import Row
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("result_analyst").getOrCreate()
item_result = spark.read.csv('../results/item_based_prediciton.csv').rdd
itemRDD = item_result.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), ratings=float(p[2])))
itemRatings = spark.createDataFrame(itemRDD)
itemRatings = itemRatings.sort(['userId', 'ratings'], ascending=False).show(200)

user_result = spark.read.csv('../results/user_based_prediciton.csv').rdd
userRDD = user_result.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), ratings=float(p[2])))
userRatings = spark.createDataFrame(userRDD)
userRatings = userRatings.sort(['userId', 'ratings'], ascending=False).show(200)
