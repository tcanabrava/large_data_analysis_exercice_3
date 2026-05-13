
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="movieplots/wiki_movie_plots_deduped.csv")
    parser.add_argument("--stopwords", default="../Data/stopwords.txt")
    parser.add_argument("--numFreq", type=int, default=5000)
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()
