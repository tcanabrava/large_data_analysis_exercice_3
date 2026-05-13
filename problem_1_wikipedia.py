import argparse
import nltk
from pyspark.sql import SparkSession
from nltk.corpus import stopwords as nltk_sw
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../Data/Wikipedia-En-41784-Articles/*/*")
    parser.add_argument("--numFreq", type=int, default=5000)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--use_nlp", action="store_true", default=False)
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()
    return args

def update_nltk_stopwords():
    for _corpus in ("stopwords", "punkt_tab", "wordnet"):
        nltk.download(_corpus, quiet=True)


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

# Reduces the words into a single common base (lemmatization)
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


def main():
    update_nltk_stopwords()
    args = parse_args()

    spark = SparkSession.builder.appName("RunLSA_Wikipedia").getOrCreate()
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

    if args.use_nlp:
        lemmatized = plainText.mapPartitions(
            lambda it: (plainTextToLemmas(x, bStopWords.value) for x in it)
        )

if __name__ == "__main__":
    main()
