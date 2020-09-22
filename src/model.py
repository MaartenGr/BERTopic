import numpy as np
import pandas as pd

import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Topic:
    def __init__(self):
        pass

    def fit(self, documents):
        # Extract embeddings
        # TO DO: Add logging through decorators
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(documents, show_progress_bar=True)

        # Reduce dimensionality
        umap_embeddings = umap.UMAP(n_neighbors=15,
                                    n_components=5,
                                    min_dist=0.0,
                                    metric='cosine').fit_transform(embeddings)
        # Cluster documents
        cluster = hdbscan.HDBSCAN(min_cluster_size=30,
                                  metric='euclidean',
                                  cluster_selection_method='eom').fit(umap_embeddings)

        # Prepare results for c-TF-IDF
        docs_df = pd.DataFrame(documents, columns=["Doc"])
        docs_df['labels'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_label = docs_df.groupby(['labels'], as_index=False).agg({'Doc': ' '.join})

        # Apply c-TF-IDF
        m = len(documents)
        tf_idf, count = c_tf_idf(docs_per_label.Doc.values, m)

        # Extract words in topics and topic size
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_label, n=20)
        count_df = docs_df.groupby(['labels']).count().sort_values("Doc", ascending=False).reset_index()

    def reduce_dimensionality(self):
        similarities = cosine_similarity(tf_idf.T)
        np.fill_diagonal(similarities, 0)

        # Extract label to merge into and from where
        count_df = docs_df.groupby(['labels']).count().sort_values("Doc", ascending=False).reset_index()
        label_to_merge = count_df.iloc[-1].labels
        label_to_merge_into = np.argmax(similarities[label_to_merge + 1]) - 1

        print(np.max(similarities[label_to_merge + 1]))

        # Adjust labels
        docs_df.loc[docs_df.labels == label_to_merge, "labels"] = label_to_merge_into
        docs_df = docs_df.sort_values("labels")
        labels = docs_df.labels.unique()
        new_labels = [i - 1 for i in range(len(labels))]
        map_labels = {label: new_label for label, new_label in zip(labels, new_labels)}
        docs_df.labels = docs_df.labels.map(map_labels)
        docs_per_label = docs_df.groupby(['labels'], as_index=False).agg({'Doc': ' '.join})

        # Calculate new topic words
        m = len(documents)
        tf_idf, count = c_tf_idf(docs_per_label.Doc.values, m)
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_label, n=20)
        count_df = docs_df.groupby(['labels']).count().sort_values("Doc", ascending=False).reset_index()


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """ Calculate a class-based TF-IDF where m is the number of total documents. """
    # Cleaner version
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.labels)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}

    return top_n_words
