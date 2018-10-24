
import os
import download
import glob

data_dir = "data/IMDB/"


data_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"



def read_text_file(path):

    with open(path, 'rt') as file:
        lines = file.readlines()
        text = " ".join(lines)

    return text


def maybe_download_and_extract():


    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


def load_data(train=True):

    train_test_path = "train" if train else "test"

    dir_base = os.path.join(data_dir, "aclImdb", train_test_path)

    path_pattern_pos = os.path.join(dir_base, "pos", "*.txt")
    path_pattern_neg = os.path.join(dir_base, "neg", "*.txt")

    paths_pos = glob.glob(path_pattern_pos)
    paths_neg = glob.glob(path_pattern_neg)

    data_pos = [read_text_file(i) for i in paths_pos]
    data_neg = [read_text_file(i) for i in paths_neg]

    x = data_pos + data_neg
    y = [1.0] * len(data_pos) + [0.0] * len(data_neg)

    return x, y
