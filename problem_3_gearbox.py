
import argparse
import math
import time

import pyspark
from pyspark.sql import SparkSession

from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import DenseVector, Vectors

import numpy as np

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

def euclidean(a, b):
    return float(np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2)))

def distToCentroid(vector, model):
    cluster = model.predict(vector)
    return euclidean(vector.toArray(), model.clusterCenters[cluster])

def calculate_k_grids(args: argparse.Namespace, normData: pyspark.RDD[DenseVector], data: pyspark.RDD[DenseVector]):
    print(f"\nRunning K-Means for k = {args.k_min} ... {args.k_max}")
    print(f"{'k':>4}  {'time_s':>8}  {'avg_dist':>12}  {'wssse':>14}")

    all_outliers = {}

    for k in range(args.k_min, args.k_max + 1):
        t0    = time.time()
        model = KMeans.train(normData, k,
                             maxIterations=20,
                             runs=1,
                             initializationMode="k-means||")
        elapsed = time.time() - t0

        # Pair each normalized vector with its original raw vector for reporting
        dist_raw = (normData
                    .zip(data)
                    .map(lambda vr: (distToCentroid(vr[0], model), vr[1])))
        dist_raw.cache()

        avg_dist = dist_raw.map(lambda x: x[0]).mean()
        wssse    = dist_raw.map(lambda x: x[0] ** 2).sum()

        print(f"{k:>4}  {elapsed:>8.1f}  {avg_dist:>12.6f}  {wssse:>14.2f}")

        # Top-N outliers for this k (highest distance to centroid)
        top = dist_raw.top(args.top_outliers, key=lambda x: x[0])
        all_outliers[k] = top
        dist_raw.unpersist()

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

    calculate_k_grids(args, normData, data)


if __name__ == "__main__":
    main()