[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/VLAC/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/BERTopic)](https://pypi.org/project/bertopic/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/BERTopic/Code%20Checks/master)](https://pypi.org/project/bertopic/)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://maartengr.github.io/BERTopic/)
[![DOI](https://zenodo.org/badge/297672263.svg)](https://zenodo.org/badge/latestdoi/297672263)


# BERTopic

<img src="images/logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. It even supports 
visualizations similar to LDAvis! 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99).

## Installation

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

To use the visualization options, install BERTopic as follows:

```bash
pip install bertopic[visualization]
```

To use Flair embeddings, install BERTopic as follows:
```bash
pip install bertopic[flair]
```

## Getting Started
For an in-depth overview of the features of `BERTopic` 
you can check the full documentation [here](https://maartengr.github.io/BERTopic/) or you can follow along 
with the Google Colab notebook [here](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing).

### Quick Start
We start by extracting topics from the well-known 20 newsgroups dataset which is comprised of english documents:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)
```

After generating topics and their probabilities, we can access the frequent topics that were generated:

```python
>>> topic_model.get_topic_freq().head()
Topic	Count
-1	7288
49	3992
30	701
27	684
11	568
```

-1 refers to all outliers and should typically be ignored. Next, let's take a look at the most 
frequent topic that was generated, `topic 49`:

```python
>>> topic_model.get_topic(49)
[('windows', 0.006152228076250982),
 ('drive', 0.004982897610645755),
 ('dos', 0.004845038866360651),
 ('file', 0.004140142872194834),
 ('disk', 0.004131678774810884),
 ('mac', 0.003624848635985097),
 ('memory', 0.0034840976976789903),
 ('software', 0.0034415334250699077),
 ('email', 0.0034239554442333257),
 ('pc', 0.003047105930670237)]
```  

**NOTE**: Use `BERTopic(language="multilingual")` to select a model that supports 50+ languages. 

### Visualize Topics
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
understanding of the topics that were extract. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
topic_model.visualize_topics()
``` 

<img src="images/topic_visualization.gif" width="60%" height="60%" align="center" />


### Embedding Models
The parameter `embedding_model` takes in a string pointing to a sentence-transformers model, 
a SentenceTransformer, or a Flair DocumentEmbedding model. 

**Sentence-Transformers**  
You can select any model from `sentence-transformers` [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

Or select a SentenceTransformer model with your own parameters:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cpu")
topic_model = BERTopic(embedding_model=sentence_model)
```

**Flair**  
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
topic_model = BERTopic(embedding_model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).

**Custom Embeddings**    
You can also use previously generated embeddings by passing it through `fit_transform()`:

```python
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs, embeddings)
```

### Overview

| Methods | Code  | 
|-----------------------|---|
| Fit the model    |  `model.fit(docs])` |
| Fit the model and predict documents    |  `model.fit_transform(docs])` |
| Predict new documents    |  `model.transform([new_doc])` |
| Access single topic   | `model.get_topic(12)`  |   
| Access all topics     |  `model.get_topics()` |
| Get topic freq    |  `model.get_topic_freq()` |
| Visualize Topics    |  `model.visualize_topics()` |
| Visualize Topic Probability Distribution    |  `model.visualize_distribution(probabilities[0])` |
| Update topic representation | `model.update_topics(docs, topics, n_gram_range=(1, 3))` |
| Reduce nr of topics | `model.reduce_topics(docs, topics, probabilities, nr_topics=30)` |
| Find topics | `model.find_topics("vehicle")` |
| Save model    |  `model.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `model.get_params()` |
 
### Citation
To cite BERTopic in your work, please use the following bibtex reference:

```bibtex
@misc{grootendorst2020bertopic,
  author       = {Maarten Grootendorst},
  title        = {BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.5.0},
  doi          = {10.5281/zenodo.4430182},
  url          = {https://doi.org/10.5281/zenodo.4430182}
}
```
