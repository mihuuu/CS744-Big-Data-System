from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sys

spark = (SparkSession
    .builder
    .appName("Task 0")
    .master("spark://master:7077")  # e.g. spark://master:7077
    .config("spark.driver.memory", "30g")   # e.g. spark properties
    .config("spark.executor.memory", "30g") # e.g. spark properties
    .config("spark.executor.cores", 5)  # e.g. spark properties
    .config("spark.task.cpus", 1)   # e.g. spark properties
    .getOrCreate())

#if len(sys.argv) != 3:
#    print("Usage: spark-submit script.py <input_path> <output_path>")
#    sys.exit(1)

input_path = "hdfs://nn:9000/export.csv"
output_path = "hdfs://nn:9000/export_sorted.csv"

df = spark.read.csv(input_path, header=True, inferSchema=True)

sorted_df = df.orderBy(col("cca2").asc(), col("timestamp").asc())

sorted_df.write.csv(output_path, header=True)

spark.stop()