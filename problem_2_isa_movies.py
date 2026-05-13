
import argparse
import time

from pyspark.sql import SparkSession
from util import (
    buildTfIdf,
    update_nltk_stopwords,
    calculateTermFreqs,
    buildRowVectors,
    multiplyByDiagonalRowMatrix,
    termsToQueryVector,
    topDocsForTermQuery,
    plainTextToLemmas,
    topTermsInTopConcepts,
    topDocsInTopConcepts
)

from pyspark.sql.types import (ArrayType, StructType, StructField,
                                IntegerType, StringType)

from pyspark.sql.functions import udf
from nltk.corpus import stopwords as nltk_sw
from pyspark.mllib.linalg.distributed import RowMatrix

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
    return args

def main():
    args = parse_args()
    spark = SparkSession.builder.appName("RunLSA_Movies").getOrCreate()
    sc    = spark.sparkContext
    sc.setLogLevel("WARN")

    update_nltk_stopwords()
    stopwords = set(nltk_sw.words("english"))
    bStopWords = sc.broadcast(stopwords)

    df = (spark.read
          .option("header",          "true")
          .option("enforceSchema",   "true")
          .option("quote",           '"')
          .option("escape",          '"')
          .option("multiLine",       "true")
          .schema(movie_csv_schema())
          .csv(args.data_path))

    df = df.select("title", "genre", "plot").na.drop(subset=["plot"])
    numDocs = df.count()
    print(f"Movie articles loaded: {numDocs}")
    df.show(5, truncate=80)

    @udf(ArrayType(StringType()))
    def lemmatize_udf(text):
        return plainTextToLemmas(("", text), bStopWords.value)[1]

    df = df.withColumn("features", lemmatize_udf(df["plot"]))
    df.cache()
    df.select("title", "features").show(5, truncate=80)

    # Build per-document term-frequency dicts from features
    featureRDD = df.rdd.map(lambda row: calculateTermFreqs((row.title, row.features)))
    featureRDD.cache()

    # Collect metadata (title + genres) keyed by row index
    docMeta = (df.rdd
               .map(lambda row: {
                   "title":  row.title or "",
                   "genres": [g.strip() for g in (row.genre or "unknown").split(",")]
               })
               .zipWithUniqueId()
               .map(lambda x: (x[1], x[0]))
               .collectAsMap())

    # ── SVD decomposition ────────────────────────────────────────────────
    print(f"\nBuilding TF-IDF: numFreq={args.numFreq} ...")
    t0 = time.time()
    idfs, idTerms, termIds, bIdfs, bIdTerms = buildTfIdf(featureRDD, args.numFreq, numDocs, sc)
    rowVectors = buildRowVectors(featureRDD, bIdTerms, bIdfs)
    rowVectors.cache()
    svd = RowMatrix(rowVectors).computeSVD(args.k, computeU=True)
    elapsed = time.time() - t0
    print(f"SVD (numFreq={args.numFreq}, k={args.k}) computed in {elapsed:.1f}s")

    numConcepts = min(args.k, 25)
    top_terms = topTermsInTopConcepts(svd, numConcepts, 25, termIds)
    top_docs  = topDocsInTopConcepts(svd, numConcepts, 25, docMeta)


    # ── (c + d) Print top terms and docs (with genres) ───────────────────────
    print(f"\n=== Top-25 terms / docs (with top-5 genres) under top-{numConcepts} concepts ===")
    for i, (terms, docs) in enumerate(zip(top_terms, top_docs)):
        print(f"\nConcept {i + 1}:")
        print("  Terms: " + ", ".join(t for t, _ in terms))
        print("  Docs:")
        for title, score, genres in docs[:5]:
            print(f"    [{score:.4f}] {title!r}  genres: {genres}")

    # ── (e) Keyword queries ──────────────────────────────────────────────────
    US = multiplyByDiagonalRowMatrix(svd.U, svd.s)

    queries = [
        ["love", "romance", "wedding"],
        ["war", "battle", "soldier"],
        ["murder", "detective", "crime"],
        ["space", "alien", "planet"],
        ["horror", "ghost", "haunted"],
        ["family", "father", "child"],
        ["comedy", "funny", "joke"],
        ["adventure", "treasure", "journey"],
        ["vampire", "blood", "monster"],
        ["robot", "artificial", "intelligence"],
    ]

    print("\n=== Movie Search Engine Queries ===")
    for q in queries:
        qvec = termsToQueryVector([t.lower() for t in q], idTerms, idfs)
        if qvec is None:
            print(f"Query {q}: no terms in vocabulary")
            continue
        results = topDocsForTermQuery(US, svd.V, qvec, docMeta, n=5)
        print(f"\nQuery: {q}")
        for rank, (title, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.6f}] {title!r}")

    spark.stop()

if __name__ == "__main__":
    main()