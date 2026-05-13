
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
import nltk
from pyspark import RDD, SparkContext

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
