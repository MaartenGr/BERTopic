import os
import json
import numpy as np

from pathlib import Path
from tempfile import TemporaryDirectory


# HuggingFace Hub
try:
    from huggingface_hub import (
        create_repo,
        get_hf_file_metadata,
        hf_hub_download,
        hf_hub_url,
        repo_type_and_id_from_hf_id,
        upload_folder,
    )

    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

# Typing
from typing import Union

# Pytorch check
try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

# Image check
try:
    from PIL import Image

    _has_vision = True
except ImportError:
    _has_vision = False

from bertopic._topics import Topics


TOPICS_NAME = "topics.json"
CONFIG_NAME = "config.json"

HF_WEIGHTS_NAME = "topic_embeddings.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "topic_embeddings.safetensors"  # safetensors version

CTFIDF_WEIGHTS_NAME = "ctfidf.bin"  # default pytorch pkl
CTFIDF_SAFE_WEIGHTS_NAME = "ctfidf.safetensors"  # safetensors version
CTFIDF_CFG_NAME = "ctfidf_config.json"

MODEL_CARD_TEMPLATE = """
---
tags:
- bertopic
library_name: bertopic
pipeline_tag: {PIPELINE_TAG}
---

# {MODEL_NAME}

This is a [BERTopic](https://github.com/MaartenGr/BERTopic) model.
BERTopic is a flexible and modular topic modeling framework that allows for the generation of easily interpretable topics from large datasets.

## Usage

To use this model, please install BERTopic:

```
pip install -U bertopic
```

You can use the model as follows:

```python
from bertopic import BERTopic
topic_model = BERTopic.load("{PATH}")

topic_model.get_topic_info()
```

## Topic overview

* Number of topics: {NR_TOPICS}
* Number of training documents: {NR_DOCUMENTS}

<details>
  <summary>Click here for an overview of all topics.</summary>

  {TOPICS}

</details>

## Training hyperparameters

{HYPERPARAMS}

## Framework versions

{FRAMEWORKS}
"""


def migrate_topics_pre_0_17_4(topics_dict: dict) -> Topics:
    """Migrate old format (<=0.17.4) topics dict to new format.

    Old format:
        {
            "topic_representations": {"-1": [["word", 0.5], ...], ...},
            "topics": [0, 1, -1, ...],
            "topic_sizes": {"-1": 100, ...},
            "topic_mapper": [[...], [...]],
            "topic_labels": {"-1": "...", ...},
            "custom_labels": null or [...],
            "_outliers": 1,
            "topic_aspects": {"Aspect": {"-1": [...], ...}, ...}
        }

    New format uses Topics.to_dict() structure.
    """
    from bertopic._topics import Topics, Topic, Keywords, TopicType, TopicAction

    # Get unique topic IDs from representations (more reliable than predictions)
    topic_ids = sorted([int(k) for k in topics_dict.get("topic_representations", {}).keys()])

    # Build Topics object
    topics = Topics()
    topics._original_predictions = np.array(topics_dict.get("topics", []))
    topics.actions = [TopicAction.INITIALIZED]

    # Reconstruct mapping from topic_mapper (2D array of mappings history)
    topic_mapper = topics_dict.get("topic_mapper")
    if topic_mapper and len(topic_mapper) > 0:
        # The last row contains the most recent mapping
        # Format: each row is [original_id_0_maps_to, original_id_1_maps_to, ...]
        # Index corresponds to original ID (offset by outliers)
        has_outliers = topics_dict.get("_outliers", 0)
        last_mapping = topic_mapper[-1] if isinstance(topic_mapper[0], list) else topic_mapper

        mapping_dict = {}
        for idx, mapped_to in enumerate(last_mapping):
            original_id = idx - has_outliers
            mapping_dict[original_id] = int(mapped_to)

        topics.mapping._mapping = mapping_dict
        topics.mapping._recent_mapping = mapping_dict.copy()

    # Custom labels (if set, these override generated labels)
    custom_labels = topics_dict.get("custom_labels")
    custom_labels_dict = {}
    if custom_labels:
        for idx, label in enumerate(custom_labels):
            topic_id = idx - topics_dict.get("_outliers", 0)
            custom_labels_dict[topic_id] = label

    # Create Topic objects
    for topic_id in topic_ids:
        str_id = str(topic_id)
        label = custom_labels_dict.get(topic_id) or topics_dict.get("topic_labels", {}).get(str_id)
        topic_type = TopicType.OUTLIER if topic_id == -1 else TopicType.NORMAL
        nr_documents = topics_dict.get("topic_sizes", {}).get(str_id, 0)

        # Main representation in <= v0.17.4 is always "Keywords"
        rep_data = topics_dict.get("topic_representations", {}).get(str_id, [])
        representations = {"Main": Keywords(data=[tuple(item) for item in rep_data])}

        # Topic aspects (additional representations)
        for aspect_name, aspect_data in topics_dict.get("topic_aspects", {}).items():
            if aspect_name == "Visual_Aspect":
                continue  # TODO: Visual aspects currently not handled due to missing image representation

            aspect_value = aspect_data[str_id]
            representations[aspect_name] = Keywords(data=[tuple(item) for item in aspect_value])

        # Create Topic
        topic = Topic(
            id=topic_id,
            _label=label,
            nr_documents=nr_documents,
            topic_type=topic_type,
            representations=representations,
            representative_documents=[],
        )

        topics.topics[topic_id] = topic

    return topics


def push_to_hf_hub(
    model,
    repo_id: str,
    commit_message: str = "Add BERTopic model",
    token: str | None = None,
    revision: str | None = None,
    private: bool = False,
    create_pr: bool = False,
    model_card: bool = True,
    serialization: str = "safetensors",
    save_embedding_model: Union[str, bool] = True,
    save_ctfidf: bool = False,
):
    """Push your BERTopic model to a HuggingFace Hub.

    Arguments:
        model: The BERTopic model to push
        repo_id: The name of your HuggingFace repository
        commit_message: A commit message
        token: Token to add if not already logged in
        revision: Repository revision
        private: Whether to create a private repository
        create_pr: Whether to upload the model as a Pull Request
        model_card: Whether to automatically create a modelcard
        serialization: The type of serialization.
                       Either `safetensors` or `pytorch`
        save_embedding_model: A pointer towards a HuggingFace model to be loaded in with
                                SentenceTransformers. E.g.,
                                `sentence-transformers/all-MiniLM-L6-v2`
        save_ctfidf: Whether to save c-TF-IDF information
    """
    if not _has_hf_hub:
        raise ValueError(
            "Make sure you have the huggingface hub installed via `pip install --upgrade huggingface_hub`"
        )

    # Create repo if it doesn't exist yet and infer complete repo_id
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Temporarily save model and push to HF
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        model.save(
            tmpdir,
            serialization=serialization,
            save_embedding_model=save_embedding_model,
            save_ctfidf=save_ctfidf,
        )

        # Add README if it does not exist
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
        except:  # noqa: E722
            if model_card:
                readme_text = generate_readme(model, repo_id)
                readme_path = Path(tmpdir) / "README.md"
                readme_path.write_text(readme_text, encoding="utf8")

        # Upload model
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def load_local_files(path):
    """Load local BERTopic files."""
    # Load json configs
    topics = load_cfg_from_json(path / TOPICS_NAME)
    params = load_cfg_from_json(path / CONFIG_NAME)

    # Load Topic Embeddings
    safetensor_path = path / HF_SAFE_WEIGHTS_NAME
    if safetensor_path.is_file():
        tensors = load_safetensors(safetensor_path)
    else:
        torch_path = path / HF_WEIGHTS_NAME
        if torch_path.is_file():
            tensors = torch.load(torch_path, map_location="cpu")
            tensors = {k: v.numpy() for k, v in tensors.items()}

    # c-TF-IDF
    try:
        ctfidf_tensors = None
        safetensor_path = path / CTFIDF_SAFE_WEIGHTS_NAME
        if safetensor_path.is_file():
            ctfidf_tensors = load_safetensors(safetensor_path)
        else:
            torch_path = path / CTFIDF_WEIGHTS_NAME
            if torch_path.is_file():
                ctfidf_tensors = torch.load(torch_path, map_location="cpu")
                ctfidf_tensors = {k: v.numpy() for k, v in ctfidf_tensors.items()}
        ctfidf_config = load_cfg_from_json(path / CTFIDF_CFG_NAME)
    except:  # noqa: E722
        ctfidf_config, ctfidf_tensors = None, None

    # Load images
    images = None
    if _has_vision:
        try:
            Image.open(path / "images/0.jpg")
            _has_images = True
        except:  # noqa: E722
            _has_images = False

        if _has_images:
            # Detect format: new format has "bertopic_version", old has "topic_representations"
            if "bertopic_version" in topics:
                topic_list = list(topics["topics"].keys())
            else:
                topic_list = list(topics["topic_representations"].keys())
            images = {}
            for topic in topic_list:
                image = Image.open(path / f"images/{topic}.jpg")
                images[int(topic)] = image

    return topics, params, tensors, ctfidf_tensors, ctfidf_config, images


def load_files_from_hf(path):
    """Load files from HuggingFace."""
    path = str(path)

    # Configs
    topics = load_cfg_from_json(hf_hub_download(path, TOPICS_NAME, revision=None))
    params = load_cfg_from_json(hf_hub_download(path, CONFIG_NAME, revision=None))

    # Topic Embeddings
    try:
        tensors = hf_hub_download(path, HF_SAFE_WEIGHTS_NAME, revision=None)
        tensors = load_safetensors(tensors)
    except:  # noqa: E722
        tensors = hf_hub_download(path, HF_WEIGHTS_NAME, revision=None)
        tensors = torch.load(tensors, map_location="cpu")

    # c-TF-IDF
    try:
        ctfidf_config = load_cfg_from_json(hf_hub_download(path, CTFIDF_CFG_NAME, revision=None))
        try:
            ctfidf_tensors = hf_hub_download(path, CTFIDF_SAFE_WEIGHTS_NAME, revision=None)
            ctfidf_tensors = load_safetensors(ctfidf_tensors)
        except:  # noqa: E722
            ctfidf_tensors = hf_hub_download(path, CTFIDF_WEIGHTS_NAME, revision=None)
            ctfidf_tensors = torch.load(ctfidf_tensors, map_location="cpu")
    except:  # noqa: E722
        ctfidf_config, ctfidf_tensors = None, None

    # Load images if they exist
    images = None
    if _has_vision:
        try:
            hf_hub_download(path, "images/0.jpg", revision=None)
            _has_images = True
        except:  # noqa: E722
            _has_images = False

        if _has_images:
            # Detect format: new format has "bertopic_version", old has "topic_representations"
            if "bertopic_version" in topics:
                topic_list = list(topics["topics"].keys())
            else:
                topic_list = list(topics["topic_representations"].keys())
            images = {}
            for topic in topic_list:
                image = Image.open(hf_hub_download(path, f"images/{topic}.jpg", revision=None))
                images[int(topic)] = image

    return topics, params, tensors, ctfidf_tensors, ctfidf_config, images


def generate_readme(model, repo_id: str):
    """Generate README for HuggingFace model card."""
    model_card = MODEL_CARD_TEMPLATE
    topic_table_head = "| Topic ID | Topic Keywords | Topic Frequency | Label | \n|----------|----------------|-----------------|-------| \n"

    # Get Statistics
    model_name = repo_id.split("/")[-1]
    params = {param: value for param, value in model.get_params().items() if "model" not in param}
    params = "\n".join([f"* {param}: {value}" for param, value in params.items()])
    topics = sorted(list(set(model.topics_)))
    nr_topics = str(len(set(model.topics_)))

    if model.topic_sizes_ is not None:
        nr_documents = str(sum(model.topic_sizes_.values()))
    else:
        nr_documents = ""

    # Topic information
    topic_keywords = [" - ".join(next(zip(*model.get_topic(topic)))[:5]) for topic in topics]
    topic_freq = [model.get_topic_freq(topic) for topic in topics]
    topic_labels = (
        model.custom_labels_ if model.custom_labels_ else [model.topic_labels_[topic] for topic in topics]
    )
    topics = [
        f"| {topic} | {topic_keywords[index]} | {topic_freq[topic]} | {topic_labels[index]} | \n"
        for index, topic in enumerate(topics)
    ]
    topics = topic_table_head + "".join(topics)
    frameworks = "\n".join([f"* {param}: {value}" for param, value in get_package_versions().items()])

    # Fill Statistics into model card
    model_card = model_card.replace("{MODEL_NAME}", model_name)
    model_card = model_card.replace("{PATH}", repo_id)
    model_card = model_card.replace("{NR_TOPICS}", nr_topics)
    model_card = model_card.replace("{TOPICS}", topics.strip())
    model_card = model_card.replace("{NR_DOCUMENTS}", nr_documents)
    model_card = model_card.replace("{HYPERPARAMS}", params)
    model_card = model_card.replace("{FRAMEWORKS}", frameworks)

    # Fill Pipeline tag
    has_visual_aspect = check_has_visual_aspect(model)
    if not has_visual_aspect:
        model_card = model_card.replace("{PIPELINE_TAG}", "text-classification")
    else:
        model_card = model_card.replace(
            "pipeline_tag: {PIPELINE_TAG}\n", ""
        )  # TODO add proper tag for this instance

    return model_card


def save_hf(model, save_directory, serialization: str):
    """Save topic embeddings, either safely (using safetensors) or using legacy pytorch."""
    tensors = np.array(model.topic_embeddings_, dtype=np.float32)

    if serialization == "safetensors":
        tensors = {"topic_embeddings": tensors}
        save_safetensors(save_directory / HF_SAFE_WEIGHTS_NAME, tensors)
    if serialization == "pytorch":
        assert _has_torch, "`pip install pytorch` to save as bin"
        tensors = {"topic_embeddings": torch.from_numpy(tensors)}
        torch.save(tensors, save_directory / HF_WEIGHTS_NAME)


def save_ctfidf(model, save_directory: str, serialization: str):
    """Save c-TF-IDF sparse matrix."""
    indptr = model.c_tf_idf_.indptr
    indices = model.c_tf_idf_.indices
    data = model.c_tf_idf_.data
    shape = np.array(model.c_tf_idf_.shape)
    diag = np.array(model.ctfidf_model._idf_diag.data)

    if serialization == "safetensors":
        tensors = {
            "indptr": indptr,
            "indices": indices,
            "data": data,
            "shape": shape,
            "diag": diag,
        }
        save_safetensors(save_directory / CTFIDF_SAFE_WEIGHTS_NAME, tensors)
    if serialization == "pytorch":
        assert _has_torch, "`pip install pytorch` to save as .bin"
        tensors = {
            "indptr": torch.from_numpy(indptr),
            "indices": torch.from_numpy(indices),
            "data": torch.from_numpy(data),
            "shape": torch.from_numpy(shape),
            "diag": torch.from_numpy(diag),
        }
        torch.save(tensors, save_directory / CTFIDF_WEIGHTS_NAME)


def save_ctfidf_config(model, path):
    """Save parameters to recreate CountVectorizer and c-TF-IDF."""
    config = {}

    # Recreate ClassTfidfTransformer
    config["ctfidf_model"] = {
        "bm25_weighting": model.ctfidf_model.bm25_weighting,
        "reduce_frequent_words": model.ctfidf_model.reduce_frequent_words,
    }

    # Recreate CountVectorizer
    cv_params = model.vectorizer_model.get_params()
    del cv_params["tokenizer"], cv_params["preprocessor"], cv_params["dtype"]
    if not isinstance(cv_params["analyzer"], str):
        del cv_params["analyzer"]

    config["vectorizer_model"] = {
        "params": cv_params,
        "vocab": model.vectorizer_model.vocabulary_,
    }

    with path.open("w") as f:
        json.dump(config, f, indent=2)


def save_config(model, path: str, embedding_model):
    """Save BERTopic configuration."""
    path = Path(path)
    params = model.get_params()
    config = {param: value for param, value in params.items() if "model" not in param}

    # Embedding model tag to be used in sentence-transformers
    if isinstance(embedding_model, str):
        config["embedding_model"] = embedding_model

    with path.open("w") as f:
        json.dump(config, f, indent=2)

    return config


def check_has_visual_aspect(model):
    """Check if model has visual aspect by inspecting _topics directly."""
    if _has_vision:
        for topic in model._topics:
            for rep in topic.representations.values():
                if hasattr(rep, "data") and isinstance(rep.data, Image.Image):
                    return True
    return False


def save_images(model, path: str):
    """Save topic images by inspecting _topics directly."""
    if _has_vision:
        # Find visual aspect name
        visual_aspect_name = None
        for topic in model._topics:
            for aspect_name, rep in topic.representations.items():
                if hasattr(rep, "data") and isinstance(rep.data, Image.Image):
                    visual_aspect_name = aspect_name
                    break
            if visual_aspect_name:
                break

        # Save images if found
        if visual_aspect_name:
            path.mkdir(exist_ok=True, parents=True)
            for topic in model._topics:
                rep = topic.representations.get(visual_aspect_name)
                if rep and hasattr(rep, "data") and isinstance(rep.data, Image.Image):
                    rep.data.save(path / f"{topic.id}.jpg")


def save_topics(model, path: str):
    """Save Topic-specific information using new Topics format."""
    path = Path(path)
    topics_dict = model._topics.to_dict()

    # Mark visual aspects (images saved separately by save_images)
    if check_has_visual_aspect(model):
        topics_dict["has_visual_aspect"] = True

    with path.open("w") as f:
        json.dump(topics_dict, f, indent=2, cls=NumpyEncoder)


def load_cfg_from_json(json_file: Union[str, os.PathLike]):
    """Load configuration from json."""
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def get_package_versions():
    """Get versions of main dependencies of BERTopic."""
    try:
        import platform
        from numpy import __version__ as np_version
        from pandas import __version__ as pandas_version
        from sklearn import __version__ as sklearn_version
        from plotly import __version__ as plotly_version

        try:
            from importlib.metadata import version

            hdbscan_version = version("hdbscan")
        except (ImportError, ModuleNotFoundError):
            hdbscan_version = None

        try:
            from umap import __version__ as umap_version
        except (ImportError, ModuleNotFoundError):
            umap_version = None

        try:
            from sentence_transformers import __version__ as sbert_version
        except (ImportError, ModuleNotFoundError):
            sbert_version = None

        try:
            from numba import __version__ as numba_version
        except (ImportError, ModuleNotFoundError):
            numba_version = None

        try:
            from transformers import __version__ as transformers_version
        except (ImportError, ModuleNotFoundError):
            transformers_version = None

        return {
            "Numpy": np_version,
            "HDBSCAN": hdbscan_version,
            "UMAP": umap_version,
            "Pandas": pandas_version,
            "Scikit-Learn": sklearn_version,
            "Sentence-transformers": sbert_version,
            "Transformers": transformers_version,
            "Numba": numba_version,
            "Plotly": plotly_version,
            "Python": platform.python_version(),
        }
    except Exception as e:
        return e


def load_safetensors(path):
    """Load safetensors and check whether it is installed."""
    try:
        import safetensors.numpy

        return safetensors.numpy.load_file(path)
    except ImportError:
        raise ValueError("`pip install safetensors` to load .safetensors")


def save_safetensors(path, tensors):
    """Save safetensors and check whether it is installed."""
    try:
        import safetensors.numpy

        safetensors.numpy.save_file(tensors, path)
    except ImportError:
        raise ValueError("`pip install safetensors` to save as .safetensors")
