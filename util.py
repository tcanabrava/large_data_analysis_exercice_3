
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
import nltk

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


def processPartitionNLP(partition, stopwords):
    _ensure_nltk_data()
    for x in partition:
        yield plainTextToLemmas(x, stopwords)

def _ensure_nltk_data():
    for corpus in ("stopwords", "punkt_tab", "wordnet"):
        nltk.download(corpus, quiet=True)
