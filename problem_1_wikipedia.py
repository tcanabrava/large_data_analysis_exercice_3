import argparse
import nltk
import time

from pyspark import RDD, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Matrix, Vectors, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix, SingularValueDecomposition

import numpy as np

from scipy.sparse import csr_matrix

from nltk.corpus import stopwords as nltk_sw

from util import lemmatize, buildTfIdf, update_nltk_stopwords

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../Data/Wikipedia-En-41784-Articles/*/*")
    parser.add_argument("--numFreq", type=int, default=5000)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--use-nlp", action="store_true", default=False)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--query", nargs="+", default=None)
    args = parser.parse_args()
    return args

def parseHeader(line) -> tuple[str, str, str]:
    try:
        s = line[line.index('id="') + 4:]
        article_id = s[:s.index('"')]
        s = s[s.index('url="') + 5:]
        url = s[:s.index('"')]
        s = s[s.index('title="') + 7:]
        title = s[:s.index('"')]
        return article_id, url, title
    except Exception:
        return "", "", ""


def parse(lines) -> list[tuple[str, str]]:
    docs = []
    title = ""
    content = ""
    for line in lines:
        try:
            if line.startswith("<doc "):
                title = parseHeader(line)[2]
                content = ""
            elif line.startswith("</doc>"):
                if title and content:
                    docs.append((title, content))
            else:
                content += line + "\n"
        except Exception:
            content = ""
    return docs

def run_grid_search(spark: SparkSession, docTermFreqs: RDD, numDocs: int, tokenizer_label: str, sc: SparkContext):
    grid_numFreqs = [5000, 10000, 20000]
    grid_ks = [25, 100, 250]
    print("\n=== Grid Search ===")
    print(f"{'numFreq':>10}  {'k':>5}  {'tokenizer':>10}  {'time_s':>8}")
    for nf in grid_numFreqs:
        for ki in grid_ks:
            _, _, _, _, elapsed = runLSA(docTermFreqs, nf, numDocs, ki, sc)
            print(f"{nf:>10}  {ki:>5}  {tokenizer_label:>10}  {elapsed:>8.1f}")
    spark.stop()
    return

def calculateTermFreqs(title_terms):
    title, terms = title_terms
    freq = {}
    for t in terms:
        freq[t] = freq.get(t, 0) + 1
    return title, freq

def buildRowVectors(docTermFreqs: RDD, bIdTerms, bIdfs):
    return docTermFreqs.map(lambda x: x[1]).map(
        lambda freq: Vectors.sparse(
            len(bIdTerms.value),
            [(bIdTerms.value[t], bIdfs.value[t] * freq[t] / sum(freq.values()))
             for t in freq if t in bIdTerms.value]
        )
    )


# Latent Semantyc Analysis
def runLSA(docTermFreqs: RDD, numTerms: int, numDocs: int, k: int, sc: SparkContext):
    t0 = time.time()
    idfs, idTerms, termIds, bIdfs, bIdTerms = buildTfIdf(docTermFreqs, numTerms, numDocs, sc)
    rowVectors = buildRowVectors(docTermFreqs, bIdTerms, bIdfs)
    rowVectors.cache()
    svd = RowMatrix(rowVectors).computeSVD(k, computeU=True)
    elapsed = time.time() - t0
    bIdfs.unpersist()
    bIdTerms.unpersist()
    return svd, idfs, idTerms, termIds, elapsed


def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: int, numTerms: int, termIds: dict[int, str]) -> list[list[tuple[str, float]]]:
    v = svd.V
    arr = v.toArray().T
    result = []
    for i in range(numConcepts):
        weights = sorted([(arr[i][tid], tid) for tid in range(v.numRows)], reverse=True)
        result.append([(termIds.get(tid, str(tid)), score) for score, tid in weights[:numTerms]])
    return result


def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: int, numDocs: int, docIds: dict[int, str]) -> list[list[tuple[str, float]]]:
    result = []
    for i in range(numConcepts):
        weights = sorted(
            svd.U.rows.map(lambda row: row.toArray()[i]).zipWithUniqueId().collect(),
            key=lambda x: -x[0]
        )
        result.append([(docIds.get(did, str(did)), score) for score, did in weights[:numDocs]])
    return result

def multiplyByDiagonalRowMatrix(mat, diag):
    s_arr = diag.toArray()
    return RowMatrix(mat.rows.map(lambda v: Vectors.dense(np.multiply(v.toArray(), s_arr))))


def termsToQueryVector(terms, idTerms, idfs):
    pairs = [(idTerms[t], idfs[t]) for t in terms if t in idTerms and t in idfs]
    if not pairs:
        return None
    indices, values = zip(*pairs)
    return csr_matrix((list(values), (list(indices), [0] * len(indices))),
                      shape=(len(idTerms), 1))


def topDocsForTermQuery(US, V, query, docIds, n=10):
    concept_vec = np.dot(V.toArray().T, query.toarray()).flatten()
    mat = Matrices.dense(len(concept_vec), 1, concept_vec)
    scores = (US.multiply(mat)
                .rows.zipWithUniqueId()
                .map(lambda x: (x[0].toArray()[0], x[1]))
                .collect())
    top = sorted(scores, key=lambda x: -x[0])[:n]
    return [(docIds.get(did, str(did)), score) for score, did in top]


def main():
    update_nltk_stopwords()
    args = parse_args()

    spark = (SparkSession.builder
             .appName("RunLSA_Wikipedia")
             .config("spark.python.worker.reuse", "true")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    stopwords = set(nltk_sw.words("english"))
    bStopWords = sc.broadcast(stopwords)

    textFiles = sc.wholeTextFiles(args.data_path)
    textFiles = textFiles.sample(False, args.sample)
    print(f"Files loaded: {textFiles.count()}")

    plainText = textFiles.flatMap(lambda x: parse(x[1].split("\n")))
    plainText.cache()
    numDocs = plainText.count()
    print(f"Articles parsed: {numDocs}")

    tokenizer_label = "NLP" if args.use_nlp else "Simple"
    print(f"Tokenizer: {tokenizer_label}")

    lemmatized = lemmatize(args, plainText, bStopWords)
    docTermFreqs = lemmatized.map(calculateTermFreqs)
    docTermFreqs.cache()
    docIds = (docTermFreqs
              .map(lambda x: x[0])
              .zipWithUniqueId()
              .map(lambda x: (x[1], x[0]))
              .collectAsMap())

    if args.grid_search:
        run_grid_search(spark, docTermFreqs, numDocs, tokenizer_label, sc)
        return


    print(f"\nRunning LSA: numFreq={args.numFreq}, k={args.k}, tokenizer={tokenizer_label}")
    svd, idfs, idTerms, termIds, elapsed = runLSA(docTermFreqs, args.numFreq, numDocs, args.k, sc)
    print(f"SVD computed in {elapsed:.1f}s")

    numConcepts = min(args.k, 25)

    # Top-25 terms and top-25 docs under top-25 concepts
    top_terms = topTermsInTopConcepts(svd, numConcepts, 25, termIds)
    top_docs  = topDocsInTopConcepts(svd, numConcepts, 25, docIds)

    print(f"\n=== Top-25 terms / docs under top-{numConcepts} concepts ===")
    for i, (terms, docs) in enumerate(zip(top_terms, top_docs)):
        print(f"\nConcept {i + 1}:")
        print("  Terms: " + ", ".join(t for t, _ in terms))
        print("  Docs:  " + ", ".join(d for d, _ in docs))

    US = multiplyByDiagonalRowMatrix(svd.U, svd.s)
    sample_queries = args.query and [args.query] or [
        ["computer", "science"],
        ["war", "battle", "army"],
        ["music", "song", "album"],
        ["physics", "quantum", "energy"],
        ["history", "ancient", "empire"],
        ["football", "soccer", "player"],
        ["film", "movie", "director"],
        ["mathematics", "algebra", "geometry"],
    ]

    print("\n=== Search Engine ===")
    for q in sample_queries:
        q_lower = [t.lower() for t in q]
        qvec = termsToQueryVector(q_lower, idTerms, idfs)
        if qvec is None:
            print(f"Query {q}: no terms found in vocabulary")
            continue
        results = topDocsForTermQuery(US, svd.V, qvec, docIds, n=10)
        print(f"\nQuery: {q}")
        for rank, (title, score) in enumerate(results, 1):
            print(f"  {rank:2}. [{score:.6f}] {title}")

    spark.stop()

if __name__ == "__main__":
    main()
