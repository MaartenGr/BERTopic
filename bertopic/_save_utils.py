import os
import sys
import json
import numpy as np

from pathlib import Path
from tempfile import TemporaryDirectory


# HuggingFace Hub
try:
    from huggingface_hub import (
        create_repo, get_hf_file_metadata,
        hf_hub_download, hf_hub_url,
        repo_type_and_id_from_hf_id, upload_folder)
    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

# Typing
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from typing import Union, Mapping, Any

# Safetensors check
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

# Pytorch check
try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

TOPICS_NAME = "topics.json"
CONFIG_NAME = "config.json"

HF_WEIGHTS_NAME = "topic_embeddings.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "topic_embeddings.safetensors"  # safetensors version

CTFIDF_WEIGHTS_NAME = "ctfidf.bin"  # default pytorch pkl
CTFIDF_SAFE_WEIGHTS_NAME = "ctfidf.safetensors"  # safetensors version
CTFIDF_CFG_NAME = "ctfidf_config.json"


def push_to_hf_hub(
        model,
        repo_id: str,
        commit_message: str = 'Add BERTopic model',
        token: str = None,
        revision: str = None,
        private: bool = False,
        create_pr: bool = False,
        model_card: Mapping[str, Any] = None,
        serialization: str = "safetensors",
        save_embedding_model: str = None,
        save_ctfidf: bool = False,
        ):
    """ Push your BERTopic model to a HuggingFace Hub

    Arguments:
        repo_id: The name of your HuggingFace repository
        commit_message: A commit message
        token: Token to add if not already logged in
        revision: Repository revision
        private: Whether to create a private repository
        create_pr: Whether to upload the model as a Pull Request
        model_card: Create a model card when creating the repository
        serialization: The type of serialization.
                       Either `safetensors` or `pytorch`
        save_embedding_model: A pointer towards a HuggingFace model to be loaded in with
                                SentenceTransformers. E.g.,
                                `sentence-transformers/all-MiniLM-L6-v2`
        save_ctfidf: Whether to save c-TF-IDF information
    """
    if not _has_hf_hub:
        raise ValueError("Make sure you have the huggingface hub installed via `pip install --upgrade huggingface_hub`")

    # Create repo if it doesn't exist yet and infer complete repo_id
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Temporarily save model and push to HF
    with TemporaryDirectory() as tmpdir:

        # Save model weights and config.
        model.save(tmpdir, serialization=serialization, save_embedding_model=save_embedding_model, save_ctfidf=save_ctfidf)

        # Add README if it does not exist
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
        except:
            model_card = model_card or {}
            model_name = repo_id.split('/')[-1]

            readme_text = generate_readme(model_card, model_name)
            readme_path = Path(tmpdir) / "README.md"
            readme_path.write_text(readme_text)

        # Upload model
        return upload_folder(repo_id=repo_id, folder_path=tmpdir, revision=revision,
                             create_pr=create_pr, commit_message=commit_message)


def load_local_files(path):
    """ Load local BERTopic files """
    # Load json configs
    topics = load_cfg_from_json(path / TOPICS_NAME)
    params = load_cfg_from_json(path / CONFIG_NAME)

    # Load Topic Embeddings
    safetensor_path = path / HF_SAFE_WEIGHTS_NAME
    if safetensor_path.is_file():
        tensors = safetensors.torch.load_file(safetensor_path, device="cpu")
    else:
        torch_path = path / HF_WEIGHTS_NAME
        if torch_path.is_file():
            tensors = torch.load(torch_path, map_location="cpu")

    # c-TF-IDF
    ctfidf_tensors = None
    safetensor_path = path / CTFIDF_SAFE_WEIGHTS_NAME
    if safetensor_path.is_file():
        ctfidf_tensors = safetensors.torch.load_file(safetensor_path, device="cpu")
    else:
        torch_path = path / CTFIDF_WEIGHTS_NAME
        if torch_path.is_file():
            ctfidf_tensors = torch.load(torch_path, map_location="cpu")
    ctfidf_config = load_cfg_from_json(path / CTFIDF_CFG_NAME)

    return topics, params, tensors, ctfidf_tensors, ctfidf_config


def load_files_from_hf(path):
    """ Load files from HuggingFace. """
    path = str(path)

    # Configs
    topics = load_cfg_from_json(hf_hub_download(path, TOPICS_NAME, revision=None))
    params = load_cfg_from_json(hf_hub_download(path, CONFIG_NAME, revision=None))

    # Topic Embeddings
    try:
        tensors = hf_hub_download(path, HF_SAFE_WEIGHTS_NAME, revision=None)
        tensors = safetensors.torch.load_file(tensors, device="cpu")
    except:
        tensors = hf_hub_download(path, HF_WEIGHTS_NAME, revision=None)
        tensors = torch.load(tensors, map_location="cpu")

    # c-TF-IDF
    try:
        ctfidf_config = load_cfg_from_json(hf_hub_download(path, CTFIDF_CFG_NAME, revision=None))
        try:
            ctfidf_tensors = hf_hub_download(path, CTFIDF_SAFE_WEIGHTS_NAME, revision=None)
            ctfidf_tensors = safetensors.torch.load_file(tensors, device="cpu")
        except:
            ctfidf_tensors = hf_hub_download(path, CTFIDF_WEIGHTS_NAME, revision=None)
            ctfidf_tensors = torch.load(tensors, map_location="cpu")
    except:
        ctfidf_config, ctfidf_tensors = None, None

    return topics, params, tensors, ctfidf_tensors, ctfidf_config


def generate_readme(model_card: dict, model_name: str):
    """ Generate README for HuggingFace model card """
    readme_text = "---\n"
    readme_text += "tags:\n- image-classification\n- bertopic\n"
    readme_text += "library_name: bertopic\n"
    readme_text += f"license: {model_card.get('license', 'mit')}\n"
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"

    if 'description' in model_card:
        readme_text += f"\n{model_card['description']}\n"

    if 'details' in model_card:
        readme_text += "\n## Model Details\n\n"
        for k, v in model_card['details'].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"

    if 'usage' in model_card:
        readme_text += "\n## Model Usage\n"
        readme_text += model_card['usage']
        readme_text += '\n'

    return readme_text


def save_hf(model, save_directory, serialization: str):
    """ Save topic embeddings, either safely (using safetensors) or using legacy pytorch """
    tensors = torch.from_numpy(np.array(model.topic_embeddings_, dtype=np.float32))
    tensors = {"topic_embeddings": tensors}

    if serialization == "safetensors":
        assert _has_safetensors, "`pip install safetensors` to use .safetensors"
        safetensors.torch.save_file(tensors, save_directory / HF_SAFE_WEIGHTS_NAME)
    if serialization == "pytorch":
        assert _has_torch, "`pip install pytorch` to save as bin"
        torch.save(tensors, save_directory / HF_WEIGHTS_NAME)


def save_ctfidf(model,
                save_directory: str,
                serialization: str):
    """ Save c-TF-IDF sparse matrix """
    indptr = torch.from_numpy(model.c_tf_idf_.indptr)
    indices = torch.from_numpy(model.c_tf_idf_.indices)
    data = torch.from_numpy(model.c_tf_idf_.data)
    shape = torch.from_numpy(np.array(model.c_tf_idf_.shape))
    diag = torch.from_numpy(np.array(model.ctfidf_model._idf_diag.data))
    tensors = {
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "shape": shape,
        "diag": diag
    }

    if serialization == "safetensors":
        assert _has_safetensors, "`pip install safetensors` to use .safetensors"
        safetensors.torch.save_file(tensors, save_directory / CTFIDF_SAFE_WEIGHTS_NAME)
    if serialization == "pytorch":
        assert _has_torch, "`pip install pytorch` to save as .bin"
        torch.save(tensors, save_directory / CTFIDF_WEIGHTS_NAME)


def save_ctfidf_config(model, path):
    """ Save parameters to recreate CountVectorizer and c-TF-IDF """
    config = {}

    # Recreate ClassTfidfTransformer
    config["ctfidf_model"] = {
        "bm25_weighting": model.ctfidf_model.bm25_weighting,
        "reduce_frequent_words": model.ctfidf_model.reduce_frequent_words
    }

    # Recreate CountVectorizer
    cv_params = model.vectorizer_model.get_params()
    del cv_params["tokenizer"], cv_params["preprocessor"], cv_params["dtype"]
    if not isinstance(cv_params["analyzer"], str):
        del cv_params["analyzer"]

    config["vectorizer_model"] = {
        "params": cv_params,
        "vocab": model.vectorizer_model.vocabulary_
    }

    with path.open('w') as f:
        json.dump(config, f, indent=2)


def save_config(model, path: str, embedding_model):
    """ Save BERTopic configuration """
    path = Path(path)
    params = model.get_params()
    config = {param: value for param, value in params.items() if "model" not in param}

    # Embedding model tag to be used in sentence-transformers
    if embedding_model:
        config["embedding_model"] = embedding_model

    with path.open('w') as f:
        json.dump(config, f, indent=2)

    return config


def save_topics(model, path: str):
    """ Save Topic-specific information """
    path = Path(path)
    topics = {
        "topic_representations": model.topic_representations_,
        "topics": [int(topic) for topic in model.topics_],
        "topic_sizes": model.topic_sizes_,
        "topic_mapper": np.array(model.topic_mapper_.mappings_, dtype=int).tolist(),
        "topic_labels": model.topic_labels_,
        "custom_labels": model.custom_labels_,
        "_outliers": int(model._outliers)
    }

    with path.open('w') as f:
        json.dump(topics, f, indent=2)


def load_cfg_from_json(json_file: Union[str, os.PathLike]):
    """ Load configuration from json """
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)
