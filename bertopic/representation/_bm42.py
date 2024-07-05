import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy
from typing import Mapping, List, Tuple
from scipy.sparse import csr_matrix, lil_matrix
from bertopic.representation._base import BaseRepresentation

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import get_device_name
from sentence_transformers.models import Transformer


class AttentionTransformer(Transformer):
    def forward(self, features):
        """Returns aggregated attention matrix."""
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        # Calculate attention matrix
        attentions = self.auto_model(**trans_features, output_attentions=True).attentions[-1]

        # Average attention weights
        weights = []
        for index in range(trans_features["input_ids"].shape[0]):
            weight = torch.mean(attentions[index, :, 0], axis=0)
            weights.append(weight)
        weights = torch.stack(weights)
        features["attention"] = weights

        # Empty intialization to prevent errors
        features["sentence_embedding"] = np.zeros((10, 10))
        return features


class BM42Inspired(BaseRepresentation):
    def __init__(
        self,
        model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_n_words: int = 10,
        nr_repr_docs: int = 50,
        nr_samples: int = 1000,
        random_state: int = 42,
        recalculate_idf: bool = False,
        device: str = None,
    ):
        """Use a BM42-like model to fine-tune the topic representations.

        The algorithm follows BM42 (https://qdrant.tech/articles/bm42/)
        but does some optimization in order to speed up inference.

        The steps are as follows. First, we extract the top n representative
        documents per topic. To extract the representative documents, we
        randomly sample a number of candidate documents per cluster
        which is controlled by the `nr_samples` parameter.

        Then, the top n representative documents are extracted by calculating
        the c-TF-IDF representation for the candidate documents.

        For all representative documents per topic, their attention matrix is
        calculated and all weights are summed. The weights are then multiplied
        by the class-based IDF values to get the final BM42 representation.

        Arguments:
            model_path: The path to the model and tokenizer to use for creating the attention matrix
            top_n_words: The top n words to extract per topic.
            nr_repr_docs: The number of representative documents to extract per cluster.
            nr_samples: The number of candidate documents to extract per cluster.
            nr_candidate_words: The number of candidate words per cluster.
            random_state: The random state for randomly sampling candidate documents.
            recalculate_idf: Whether to recalculate the IDF values for the BM42 representation
                             based on the most representative documents or use the existing IDF
                             values as calculated over the entire corpus.
            device: The device to run the model on. If None, it will use the GPU if available.

        Usage:

        ```python
        from bertopic.representation import BM42Inspired
        from bertopic import BERTopic

        # Create your representation model
        representation_model = BM42Inspired("sentence-transformers/all-MiniLM-L6-v2")

        # Use the representation model in BERTopic on top of the default pipeline
        topic_model = BERTopic(representation_model=representation_model)
        ```
        """
        self.model_path = model_path
        self.top_n_words = top_n_words
        self.nr_repr_docs = nr_repr_docs
        self.nr_samples = nr_samples
        self.recalculate_idf = recalculate_idf
        self.random_state = random_state

        # Create tokenizer and model
        if device:
            self.device = device
        else:
            self.device = get_device_name()
        module = AttentionTransformer(self.model_path)
        self.embedding_model = SentenceTransformer(modules=[module])

        # Initialize private variables for easier tracking
        self._vocab = None
        self._idf = None

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # We extract the top n representative documents per class
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, self.nr_samples, self.nr_repr_docs
        )
        topic_labels = list(topics.keys())

        # Extract the weights for the representative documents
        if not self.recalculate_idf:
            updated_topics = {}
            for topic, documents in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
                bow = self._extract_weights(documents)
                bm42 = self._apply_idf(bow, topic_model)
                updated_topics[topic] = bm42
        else:
            bows = []
            for topic, documents in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
                bow = self._extract_weights(documents)
                bows.append(bow)
            updated_topics = self._calculate_and_apply_idf(bows, topic_model, topic_labels)

        return updated_topics

    def _extract_weights(self, documents: List[str]) -> Mapping[str, float]:
        """Extract the weights for the documents using the attention matrix.

        Arguments:
            documents: List of documents
        """
        bow = {}
        attention = self.embedding_model.encode(documents, output_value="attention", show_progress_bar=False)

        for index, doc in enumerate(documents):
            tokens = self.embedding_model.tokenizer.tokenize(doc)
            weights = attention[index][1 : 1 + len(tokens)].cpu().detach().tolist()
            combined_tokens, combined_weights = tokens_to_words(tokens, weights)

            # Summarize weights
            for token_id, token in enumerate(combined_tokens):
                if bow.get(token):
                    bow[token] += combined_weights[token_id]
                else:
                    bow[token] = combined_weights[token_id]

        return bow

    def _apply_idf(self, bow: Mapping[str, float], topic_model) -> List[Tuple[str, float]]:
        """Apply the inverse document frequency to the attention weights.

        Arguments:
            bow: The bag of words representation of the representative documents
                 in the topic
            topic_model: The BERTopic model
        """
        if self._vocab is None:
            vocab = list(topic_model.vectorizer_model.get_feature_names_out())
            self._vocab = {word: index for index, word in enumerate(vocab)}

        if self._idf is None:
            self._idf = topic_model.ctfidf_model.idf_

        # Apply IDF to the attention weights
        weighted_bow = {}
        for word, value in bow.items():
            if self._vocab.get(word):
                index = self._vocab[word]
                idf = self._idf[index]
                weighted_bow[word] = value * idf

        # Order by value
        bm42 = pd.DataFrame.from_dict(weighted_bow, orient="index", columns=["value"])
        bm42 = bm42.sort_values("value", ascending=False).head(self.top_n_words).reset_index()
        bm42 = [tuple(vals) for vals in bm42.values]
        return bm42

    def _calculate_and_apply_idf(
        self,
        bows: List[Mapping[str, float]],
        topic_model,
        topic_labels: List[str],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Calculate and apply the inverse document frequency to the attention weights.

        Arguments:
            bows: The bag of words representation of the representative documents
                  in the topic
            topic_model: The BERTopic model
            topic_labels: The topic labels
        """
        unique_words = sorted({word for entry in bows for word in entry})
        word_index = {word: i for i, word in enumerate(unique_words)}

        # Convert values to sparse matrix
        sparse_matrix = lil_matrix((len(bows), len(unique_words)), dtype=np.float32)
        for i, entry in enumerate(bows):
            for word, value in entry.items():
                sparse_matrix[i, word_index[word]] = value
        sparse_matrix = sparse_matrix.tocsr()

        # Fit c-TF-IDF
        ctfidf_model = deepcopy(topic_model.ctfidf_model)
        ctfidf_model.reduce_frequent_words = True
        ctfidf_model = ctfidf_model.fit(sparse_matrix, multiplier=None)
        c_tf_idf = ctfidf_model.transform(sparse_matrix)

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        words = unique_words
        top_n_words = max(topic_model.top_n_words, self.top_n_words)
        indices = topic_model._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = topic_model._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top n words per topic based on c-TF-IDF score
        updated_topics = {
            label: [
                (words[word_index], score) if word_index is not None and score > 0 else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(topic_labels)
        }
        return updated_topics


def tokens_to_words(tokens: List[str], weights: List[float]) -> Tuple[List[str], List[float]]:
    """Convert tokens to words by combining subwords and summing their weights.

    Arguments:
        tokens: List of tokens
        weights: List of weights
    """
    combined_tokens = []
    combined_weights = []

    current_token = ""
    current_weight = 0

    for token, weight in zip(tokens, weights):
        if "[" not in token:
            if token.startswith("##"):
                current_token += token[2:]
                current_weight += weight
            else:
                if current_token:  # add the previous token to the list
                    combined_tokens.append(current_token)
                    combined_weights.append(current_weight)
                current_token = token
                current_weight = weight

    # Add the last token to the list
    if current_token:
        combined_tokens.append(current_token)
        combined_weights.append(current_weight)
    return combined_tokens, combined_weights
