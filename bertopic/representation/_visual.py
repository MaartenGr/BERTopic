import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Union
from transformers.pipelines import Pipeline, pipeline

from bertopic.representation._mmr import mmr
from bertopic.representation._base import BaseRepresentation


class VisualRepresentation(BaseRepresentation):
    """ From a collection of representative documents, extract 
    images to represent topics. These topics are represented by a
    collage of images. 
    
    Arguments:
        nr_repr_images: Number of representative images to extract
        nr_samples: The number of candidate documents to extract per cluster.
        image_size: The size of the collage for each topic (width x height)

    Usage:

    ```python
    from bertopic.representation import VisualRepresentation
    from bertopic import BERTopic

    # The visual representation is typically not a core representation
    # and is advised to pass to BERTopic as an additional aspect.
    # Aspects can be labeled with dictionaries as shown below:
    representation_model = {
        "Visual_Aspect": VisualRepresentation()
    }

    # Use the representation model in BERTopic as a separate aspect
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """
    def __init__(self,
                 nr_repr_images: int = 9,
                 nr_samples: int = 500,
                 image_size: Tuple[int, int] = (600, 600),
                 text_to_image_model: Union[str, Pipeline] = None,
                 batch_size: int = 32):
        self.nr_repr_images = nr_repr_images
        self.nr_samples = nr_samples
        self.image_size = image_size

        # Text-to-image model
        if isinstance(text_to_image_model, Pipeline):
            self.text_to_image_model = text_to_image_model
        elif isinstance(text_to_image_model, str):
            self.text_to_image_model = pipeline("image-to-text", model=text_to_image_model)
        else:
            raise ValueError("Please select a correct transformers pipeline. For example:"
                             "pipeline('image-to-text', model='nlpconnect/vit-gpt2-image-captioning')")
        self.batch_size = batch_size

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            representative_images: Representative images per topic
        """
        # Extract image ids of most representative documents
        images = documents["Image"].values.tolist()
        _, _, _, repr_docs_ids = topic_model._extract_representative_docs(c_tf_idf, 
                                                                   documents, 
                                                                   topics,
                                                                   nr_samples=self.nr_samples,
                                                                   nr_repr_docs=self.nr_repr_images)
        unique_topics = sorted(list(topics.keys()))

        # Combine representative images into a single representation
        representative_images = {}
        for topic in tqdm(unique_topics):
            
            # Get and order represetnative images
            sliced_examplars = repr_docs_ids[topic+topic_model._outliers]
            sliced_examplars = [sliced_examplars[i:i + 3] for i in range(0, len(sliced_examplars), 3)]
            images_to_combine = [
                    [Image.open(images[index]) if isinstance(images[index], str) else images[index] for index in sub_indices]
                for sub_indices in sliced_examplars
            ]

            # Concatenate representative images
            representative_image = get_concat_tile_resize(images_to_combine, self.image_size)
            representative_images[topic] = representative_image

            # Make sure to properly close images
            if isinstance(images[0], str):
                for image_list in images_to_combine:
                    for image in image_list:
                        image.close()
        
        return representative_images
    
    def _convert_image_to_text(self,
                      images: List[str],
                      verbose: bool = False) -> List[str]:
        """ Convert a list of images to captions. 

        Arguments:
            images: A list of images or words to be converted to text.
            verbose: Controls the verbosity of the process

        Returns:
            List of captions
        """
        # Batch-wise image conversion
        if self.batch_size is not None:
            documents = []
            for batch in tqdm(self._chunks(images), disable=not verbose):
                outputs = self.text_to_image_model(batch)
                captions = [output[0]["generated_text"] for output in outputs]
                documents.extend(captions)

        # Convert images to text
        else:
            outputs = self.text_to_image_model(images)
            documents = [output[0]["generated_text"] for output in outputs]

        return documents
    
    def image_to_text(self, documents: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """ Convert images to text """
        # Create image topic embeddings
        topics = documents.Topic.values.tolist()
        images = documents.Image.values.tolist()
        df = pd.DataFrame(np.hstack([np.array(topics).reshape(-1, 1), embeddings]))
        image_topic_embeddings = df.groupby(0).mean().values
        
        # Extract image centroids
        image_centroids = {}
        unique_topics = sorted(list(set(topics)))
        for topic, topic_embedding in zip(unique_topics, image_topic_embeddings):
            indices = np.array([index for index, t in enumerate(topics) if t == topic])
            indices = mmr(topic_embedding.reshape(1, -1), embeddings[indices], indices, top_n=self.nr_repr_images, diversity=0.1)
            image_centroids[topic] = indices
            
        # Extract documents
        from tqdm import tqdm
        documents = pd.DataFrame(columns=["Document", "ID", "Topic", "Image"])
        current_id = 0
        for topic, image_ids in tqdm(image_centroids.items()):
            selected_images = [Image.open(images[index]) if isinstance(images[index], str) else images[index] for index in image_ids]
            text = self._convert_image_to_text(selected_images)
            
            for doc, image_id in zip(text, image_ids):
                documents.loc[len(documents), :] = [doc, current_id, topic, images[image_id]]
                current_id += 1
            
            # Properly close images
            if isinstance(images[image_ids[0]], str):
                for image in selected_images:
                    image.close()

        return documents

    def _chunks(self, images):     
        for i in range(0, len(images), self.batch_size):
            yield images[i:i + self.batch_size]
    

def get_concat_h_multi_resize(im_list):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    min_height = min(im.height for im in im_list)
    im_list_resize = []
    for im in im_list:
        im.resize((int(im.width * min_height / im.height), min_height), resample=0)
        im_list_resize.append(im)

    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def get_concat_v_multi_resize(im_list):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=0)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def get_concat_tile_resize(im_list_2d, image_size=(600, 600)):
    """
    Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/
    """
    im_list_v = [get_concat_h_multi_resize(im_list_h) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v).resize(image_size)
