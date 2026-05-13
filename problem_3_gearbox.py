
import argparse
import math

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


def computeStats(data):
    """Compute per-column mean and standard deviation."""
    arr_rdd   = data.map(lambda v: v.toArray())
    n         = arr_rdd.count()
    num_cols  = len(arr_rdd.first())

    sums   = arr_rdd.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    sum_sq = arr_rdd.aggregate(
        [0.0] * num_cols,
        lambda acc, v: [a + x * x for a, x in zip(acc, v)],
        lambda a, b:   [x + y     for x, y in zip(a, b)]
    )
    means  = [s / n for s in sums]
    stdevs = [math.sqrt(max(n * sq - s * s, 0.0)) / n
              for sq, s in zip(sum_sq, sums)]
    return means, stdevs


def normalizeVector(vec, means, stdevs):
    arr = vec.toArray()
    return Vectors.dense([
        (v - m) / s if s > 0 else v - m
        for v, m, s in zip(arr, means, stdevs)
    ])


def main():
    args = parse_args()

    spark = SparkSession.builder.appName("RunKMeans_Gearbox").getOrCreate()
    sc    = spark.sparkContext
    sc.setLogLevel("WARN")

    print(f"Loading gearbox data from: {args.data_path}")
    raw = sc.textFile(args.data_path)
    data = raw.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(",")]))
    data.cache()
    total = data.count()
    print(f"Total readings: {total:,}")

    print("Computing normalization statistics ...")
    means, stdevs = computeStats(data)
    print(f"  Column means:  {[round(m, 6) for m in means]}")
    print(f"  Column stdevs: {[round(s, 6) for s in stdevs]}")

    normData = data.map(lambda v: normalizeVector(v, means, stdevs))
    normData.cache()


if __name__ == "__main__":
    main()