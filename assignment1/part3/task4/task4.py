from pyspark.sql import SparkSession
import sys

if len(sys.argv) < 4:
    print("Please enter input and output file paths, and taskname")
    sys.exit(1)

input_path = "hdfs://nn:9000/" + sys.argv[1]
output_path = "hdfs://nn:9000/" + sys.argv[2]
partition = 100
task_name = sys.argv[3]

spark = (SparkSession
        .builder
        .appName(task_name)
        .master("spark://master:7077")
        .config("spark.driver.memory", "30g")
        .config("spark.executor.memory", "30g")
        .config("spark.executor.cores", 5)
        .config("spark.task.cpus", 1)
        .config("spark.local.dir", "/data/tmp")
        .getOrCreate())

sc = spark.sparkContext

#  Read file and parse edges, skip comment lines.
#  Each edge: (fromNode, toNode)
lines = sc.textFile(input_path)
edges = (lines
         .filter(lambda line: not line.startswith('#'))
         .map(lambda line: line.split("\t"))
         .map(lambda nodes: (nodes[0], nodes[1]))
        )

# source node => (node, [neighbors...])
links = edges.groupByKey().repartition(partition).cache()

# initialize ranks of each node
ranks = links.mapValues(lambda n: 1.0)

NUM_ITERATIONS = 10

def calc_rank(node):
        _node, (neighbors, rank) = node
        # contrib = rank / len(neighbors)
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            return []
        return [(neighbor, rank / num_neighbors) for neighbor in neighbors]

# run page rank for 10
for i in range(NUM_ITERATIONS):
    contributions = links.join(ranks).flatMap(lambda n: calc_rank(n))
    ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(lambda rank: 0.15 + 0.85 * rank)

ranks.saveAsTextFile(output_path)