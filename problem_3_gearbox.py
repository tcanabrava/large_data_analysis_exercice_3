
import argparse
from pyspark.sql import SparkSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default="gearbox/*.csv")
    parser.add_argument("--k_min",        type=int,   default=2)
    parser.add_argument("--k_max",        type=int,   default=12)
    parser.add_argument("--top_outliers", type=int,   default=25)
    parser.add_argument("--viz_sample",   type=float, default=0.0005)
    parser.add_argument("--output_dir",   default="./output_p3")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    spark = SparkSession.builder.appName("RunKMeans_Gearbox").getOrCreate()
    sc    = spark.sparkContext
    sc.setLogLevel("WARN")

if __name__ == "__main__":
    main()