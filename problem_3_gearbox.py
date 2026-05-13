
import argparse
import math
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyspark
import os

from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import DenseVector, Vectors

matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",    default="gearbox/*.csv")
    parser.add_argument("--k-min",        type=int,   default=2)
    parser.add_argument("--k-max",        type=int,   default=12)
    parser.add_argument("--top-outliers", type=int,   default=25)
    parser.add_argument("--viz-sample",   type=float, default=0.0005)
    parser.add_argument("--output-dir",   default="./output_p3")
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

    return all_outliers

def display_outliners(args: argparse.Namespace, all_outliers: dict):
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n=== Top-{args.top_outliers} outliers for k={k} ===")
        for rank, (dist, vec) in enumerate(all_outliers[k], 1):
            vals = ", ".join(f"{v:.6f}" for v in vec.toArray())
            print(f"  {rank:2}. dist={dist:.6f}  raw=[{vals}]")


def saveVisualization(sample_points, out_dir):
    if not sample_points:
        return

    os.makedirs(out_dir, exist_ok=True)
    data_repr = repr([[float(x) for x in pt] for pt in sample_points])

    sample = {data_repr}
    arr = np.array(sample)
    n   = len(arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Gearbox Sensor Readings — Normalized Sample (n={n})')

    pairs = [(0, 1, 'Sensor 1', 'Sensor 2'),
            (0, 2, 'Sensor 1', 'Sensor 3'),
            (1, 2, 'Sensor 2', 'Sensor 3')]

    for ax, (xi, yi, xl, yl) in zip(axes, pairs):
        ax.scatter(arr[:, xi], arr[:, yi], s=1, alpha=0.3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'{xl} vs {yl}')

    plt.tight_layout()
    out = 'gearbox_sample.png'
    plt.savefig(out, dpi=150)
    plt.close()

    print(f'Saved: {out}')

def main():
    args = parse_args()

    spark = (SparkSession.builder
             .appName("RunKMeans_Gearbox")
             .config("spark.python.worker.reuse", "true")
             .getOrCreate())
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

    all_outliers = calculate_k_grids(args, normData, data)
    display_outliners(args, all_outliers)

    print(f"\nSampling {args.viz_sample * 100:.2f}% of normalized data for visualization ...")
    sample = normData.sample(False, args.viz_sample).collect()
    print(f"Sample size: {len(sample)}")

    saveVisualization([v.toArray() for v in sample], args.output_dir)

    spark.stop()

if __name__ == "__main__":
    main()
