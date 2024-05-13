from pyspark import SparkContext, SparkConf

DATA = "./data/*.txt"
OUTPUT_DIR = "counts_filtered"

def word_count():
    sc = SparkContext.getOrCreate(SparkConf().setAppName("Word Count Example").setMaster("local"))
    try:
        textFile = sc.textFile(DATA)
        counts = (textFile
                  .flatMap(lambda line: line.split(" "))
                  .map(lambda word: (word, 1))
                  .reduceByKey(lambda a, b: a + b)
                  .filter(lambda pair: pair[1] >= 3))
        counts.saveAsTextFile(OUTPUT_DIR)
        print("Number of partitions: ", textFile.getNumPartitions())
    finally:
        sc.stop()  # Properly stop the SparkContext to free resources

word_count()