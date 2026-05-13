import argparse
import nltk
from pyspark.sql import SparkSession
from nltk.corpus import stopwords as nltk_sw

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

if __name__ == "__main__":
    main()
