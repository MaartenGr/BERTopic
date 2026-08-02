import numpy as np
from transformers import pipeline
from transformers.pipelines.base import Pipeline
from scipy.sparse import csr_matrix
from typing import Any
from bertopic.representation._base import BaseRepresentation
from bertopic._topics import Keywords
from bertopic._corpus import Corpus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


class ZeroShotClassification(BaseRepresentation):
    """Zero-shot Classification on topic keywords with candidate labels.

    Arguments:
        candidate_topics: A list of labels to assign to the topics if they
                          exceed `min_prob`
        model: A transformers pipeline that should be initialized as
               "zero-shot-classification". For example,
               `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")`
        pipeline_kwargs: Kwargs that you can pass to the transformers.pipeline
                         when it is called. NOTE: Use `{"multi_label": True}`
                         to extract multiple labels for each topic.
        min_prob: The minimum probability to assign a candidate label to a topic

    Usage:

    ```python
    from bertopic.representation import ZeroShotClassification
    from bertopic import BERTopic

    # Create your representation model
    candidate_topics = ["space and nasa", "bicycles", "sports"]
    representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    def __init__(
        self,
        candidate_topics: list[str],
        model: str = "facebook/bart-large-mnli",
        pipeline_kwargs: dict[str, Any] = {},
        min_prob: float = 0.8,
    ):
        self.candidate_topics = candidate_topics
        if isinstance(model, str):
            self.model = pipeline("zero-shot-classification", model=model)
        elif isinstance(model, Pipeline):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the HF model that you"
                "pass is either a string referring to a"
                "HF model or a `transformers.pipeline` object."
            )
        self.pipeline_kwargs = pipeline_kwargs
        self.min_prob = min_prob

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray = None,
    ) -> dict[int, Keywords]:
        """Extract topics.

        Arguments:
            topic_model: Not used
            corpus: Not used
            topic_representations: The candidate topic representations
            c_tf_idf: Not used
            embeddings: Not used

        Returns:
            updated_topics: Updated topic representations
        """
        # Classify topics
        topic_descriptions = [
            " ".join(topic_representations[topic].words) for topic in topic_representations.keys()
        ]
        classifications = self.model(topic_descriptions, self.candidate_topics, **self.pipeline_kwargs)

        # Extract labels
        updated_topics = {}
        for topic, classification in zip(topic_representations.keys(), classifications):
            topic_description = []

            # Multi-label assignment
            if self.pipeline_kwargs.get("multi_label"):
                for label, score in zip(classification["labels"], classification["scores"]):
                    if score > self.min_prob:
                        topic_description.append((label, score))

            # Single label assignment
            elif classification["scores"][0] > self.min_prob:
                topic_description = [(classification["labels"][0], classification["scores"][0])]

            # Fall back to original representation if no labels passed the threshold
            if len(topic_description) == 0:
                updated_topics[topic] = topic_representations[topic]
            else:
                if len(topic_description) < 10:
                    topic_description += [("", 0) for _ in range(10 - len(topic_description))]
                updated_topics[topic] = Keywords(topic_description)

        return updated_topics
