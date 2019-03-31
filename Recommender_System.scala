// Importing
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS
// Creating Spark Session
val spark = SparkSession.builder.appName("Recommender System").getOrCreate()
// Reading the csv file
val ratings = spark.read.option("header","true").option("inferschema","true").csv("movie_ratings.csv")
// Print the dataset
ratings.head()
ratings.printSchema()
// Splitting the dataset
val Array(training,test) = ratings.randomSplit(Array(0.8,0.2))
// Creating ALS object
val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
// Train the model on training data
val model = als.fit(training)
// Test the model
val predictions = model.transform(test)
predictions.show()

// Calculating the error
import org.apache.spark.sql.functions._
val error = predictions.select(abs($"rating"-$"prediction"))
error.na.drop().describe().show()
