import argparse
import nltk

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

def main():
    update_nltk_stopwords()
    args = parse_args()

if __name__ == "__main__":
    main()
