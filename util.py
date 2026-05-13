
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
import nltk
from pyspark import RDD, SparkContext
from pyspark.mllib.linalg import Vectors, Matrix
from pyspark.mllib.linalg.distributed import RowMatrix, SingularValueDecomposition

import math
import operator

def plainTextToLemmas(title_text, stopwords):
    title, text = title_text
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for sentence in sent_tokenize(text):
        for token in word_tokenize(sentence):
            lemma = lemmatizer.lemmatize(token.lower())
            if len(lemma) > 2 and lemma not in stopwords and lemma.isalpha():
                lemmas.append(lemma)
    return title, lemmas

def plainTextToTokens(title_text, stopwords):
    title, text = title_text
    tokens = [w.lower() for w in text.split() if len(w) > 2 and w.isalpha() and w.lower() not in stopwords]
    return title, tokens


def lemmatize(args, plainText, bStopWords):
    if not args.use_nlp:
        return plainText.map(lambda x: plainTextToTokens(
            x, bStopWords.value)
        )

    return plainText.mapPartitions(
        lambda it: processPartitionNLP(it, bStopWords.value)
    )

def _term_count(term: str) -> tuple[str, int]:
    return term, 1

def buildTfIdf(docTermFreqs: RDD, numTerms: int, numDocs: int, sc: SparkContext):
    docFreqs = (docTermFreqs
                .flatMap(lambda x: x[1].keys())
                .map(_term_count)
                .reduceByKey(operator.add, numPartitions=24))
    topDocFreqs = docFreqs.top(numTerms, key=lambda x: x[1])  # pyright: ignore[reportArgumentType]
    idfs = {term: math.log(numDocs / count) for term, count in topDocFreqs}
    idTerms = {term: i for i, (term, _) in enumerate(topDocFreqs)}
    termIds = {v: k for k, v in idTerms.items()}
    bIdfs = sc.broadcast(idfs)
    bIdTerms = sc.broadcast(idTerms)
    return idfs, idTerms, termIds, bIdfs, bIdTerms

def processPartitionNLP(partition, stopwords):
    _ensure_nltk_data()
    for x in partition:
        yield plainTextToLemmas(x, stopwords)

def _ensure_nltk_data():
    for corpus in ("stopwords", "punkt_tab", "wordnet"):
        nltk.download(corpus, quiet=True)

def update_nltk_stopwords():
    for _corpus in ("stopwords", "punkt_tab", "wordnet"):
        nltk.download(_corpus, quiet=True)

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