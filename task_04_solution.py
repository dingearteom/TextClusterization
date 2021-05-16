# coding: utf-8
"""
    Задание на кластеризацию; трюки могут быть сколь угодно грязными --
    а вот код должен быть чистым и прокомментированным предельно ясно.
"""
import nltk
import pandas as pd
from typing import List
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn import cluster

nltk.download("stopwords")

STOPS = set(stopwords.words("russian"))


class TextsPairClassifier(object):

    def __init__(self, data: List[str]):
        # get frequent words
        freq_list = set([line.strip().split(" ")[2]
                         for line in open("data/lemma.num", "r+", encoding="cp1251").readlines()
                         if line.strip()]).difference(STOPS)

        tokens = []
        raw_vectors = []

        # extract word embeddings
        with open("data/182/model.txt", "r+", encoding="utf-8") as rf:
            next(rf)
            for line in tqdm(rf):
                line = line.strip()
                splitted = line.split(" ")
                vector = np.array([float(n) for n in splitted[1:]])
                token = splitted[0].split("_")[0]

                if token in freq_list:
                    tokens.append(token)
                    raw_vectors.append(vector)

        token2id = {t: i for i, t in enumerate(tokens)}
        vectors = np.array(raw_vectors)

        # build docs embeddings
        docEmbeddings = []
        for i in range(len(data)):
            embedding = np.zeros_like(vectors[0])
            cnt = 0
            for word in data[i].split(' '):
                if word in token2id:
                    embedding += vectors[token2id[word]]
                    cnt += 1
            embedding /= cnt
            docEmbeddings.append(embedding)

        scaler = StandardScaler()
        docEmbeddings = scaler.fit_transform(docEmbeddings)

        # clusterization
        n_clusters = 7
        k_means = cluster.KMeans(n_clusters=n_clusters, random_state=10)
        k_means.fit(docEmbeddings)

        self.my_labels = k_means.labels_
        map_union_clusters = [0, 1, 2, 3, 3, 3, 0]
        for i in range(len(self.my_labels)):
            self.my_labels[i] = map_union_clusters[self.my_labels[i]]

    def label(self, id1: int, id2: int):
        """ If the items are in the same cluster, return 1, else 0; use self.pair_labels"""
        return int(self.my_labels[id1 - 1] == self.my_labels[id2 - 1])


def generate_submission():

    # reading data
    texts = pd.read_csv("data/normalized_texts.csv", index_col="id", encoding="utf-8")
    pairs = pd.read_csv("data/pairs.csv", index_col="id")

    # preparing clusters on object creation and initialization
    classifier = TextsPairClassifier(texts["paragraph_lemmatized"].to_list())

    # generating submission
    with open("data/submission.csv", "w", encoding="utf-8") as output:
        output.write("id,gold\n")
        for index, id1, id2 in pairs.itertuples():
            result = classifier.label(id1, id2)
            output.write("%s,%s\n" % (index, result))


if __name__ == "__main__":
    generate_submission()
