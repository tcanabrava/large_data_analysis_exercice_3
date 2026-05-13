
import argparse
from pyspark.sql import SparkSession
from util import update_nltk_stopwords

from pyspark.sql.types import (StructType, StructField,
                                IntegerType, StringType)

def movie_csv_schema():
    return StructType([
        StructField("release_year", IntegerType(), True),
        StructField("title",        StringType(),  True),
        StructField("origin",       StringType(),  True),
        StructField("director",     StringType(),  True),
        StructField("cast",         StringType(),  True),
        StructField("genre",        StringType(),  True),
        StructField("wiki_page",    StringType(),  True),
        StructField("plot",         StringType(),  True),
    ])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="movieplots/wiki_movie_plots_deduped.csv")
    parser.add_argument("--stopwords", default="../Data/stopwords.txt")
    parser.add_argument("--numFreq", type=int, default=5000)
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("RunLSA_Movies").getOrCreate()
    sc    = spark.sparkContext
    sc.setLogLevel("WARN")

    update_nltk_stopwords()

    df = (spark.read
          .option("header",    "true")
          .option("quote",     '"')
          .option("escape",    '"')
          .option("multiLine", "true")
          .schema(movie_csv_schema())
          .csv(args.data_path))

    df = df.select("title", "genre", "plot").na.drop(subset=["plot"])
    numDocs = df.count()
    print(f"Movie articles loaded: {numDocs}")
    df.show(5, truncate=80)
