[![PyPI - Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://pypi.org/project/bertopic/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/BERTopic/Code%20Checks/master)](https://pypi.org/project/bertopic/)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://maartengr.github.io/BERTopic/)
[![PyPI - PyPi](https://img.shields.io/pypi/v/BERTopic)](https://pypi.org/project/bertopic/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/VLAC/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2203.05794-<COLOR>.svg)](https://arxiv.org/abs/2203.05794)


# BERTopic

<img src="images/logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions.

BERTopic supports 
[**guided**](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html), 
(semi-) [**supervised**](https://maartengr.github.io/BERTopic/getting_started/supervised/supervised.html), 
and [**dynamic**](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html) topic modeling. It even supports visualizations similar to LDAvis!

Corresponding medium posts can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99) 
and [here](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8?sk=03c2168e9e74b6bda2a1f3ed953427e4). For a more detailed overview, you can read the [paper](https://arxiv.org/abs/2203.05794). 

## Installation

Installation, with sentence-transformers, can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

You may want to install more depending on the transformers and language backends that you will be using. 
The possible installations are: 

```bash
pip install bertopic[flair]
pip install bertopic[gensim]
pip install bertopic[spacy]
pip install bertopic[use]
```

## Getting Started
For an in-depth overview of the features of BERTopic 
you can check the full documentation [here](https://maartengr.github.io/BERTopic/) or you can follow along 
with one of the examples below:

| Name  | Link  |
|---|---|
| Topic Modeling with BERTopic  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)  |
| (Custom) Embedding Models in BERTopic  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18arPPe50szvcCp_Y6xS56H2tY0m-RLqv?usp=sharing) |
| Advanced Customization in BERTopic  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ClTYut039t-LDtlcd-oQAdXWgcsSGTw9?usp=sharing) |
| (semi-)Supervised Topic Modeling with BERTopic  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bxizKzv5vfxJEB29sntU__ZC7PBSIPaQ?usp=sharing)  |
| Dynamic Topic Modeling with Trump's Tweets  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1un8ooI-7ZNlRoK0maVkYhmNRl0XGK88f?usp=sharing)  |
| Topic Modeling arXiv Abstracts | [![Kaggle](https://img.shields.io/static/v1?style=for-the-badge&message=Kaggle&color=222222&logo=Kaggle&logoColor=20BEFF&label=)](https://www.kaggle.com/maartengr/topic-modeling-arxiv-abstract-with-bertopic) |


## Quick Start
We start by extracting topics from the well-known 20 newsgroups dataset containing English documents:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
```

After generating topics and their probabilities, we can access the frequent topics that were generated:

```python
>>> topic_model.get_topic_info()

Topic	Count	Name
-1	4630	-1_can_your_will_any
0	693	49_windows_drive_dos_file
1	466	32_jesus_bible_christian_faith
2	441	2_space_launch_orbit_lunar
3	381	22_key_encryption_keys_encrypted
```

-1 refers to all outliers and should typically be ignored. Next, let's take a look at the most 
frequent topic that was generated, topic 0:

```python
>>> topic_model.get_topic(0)

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

## Visualize Topics
After having trained our BERTopic model, we can iteratively go through hundreds of topics to get a good 
understanding of the topics that were extracted. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
topic_model.visualize_topics()
``` 

<img src="images/topic_visualization.gif" width="60%" height="60%" align="center" />

We can create an overview of the most frequent topics in a way that they are easily interpretable. 
Horizontal barcharts typically convey information rather well and allow for an intuitive representation 
of the topics: 

```python
topic_model.visualize_barchart()
``` 

<img src="images/topics.png" width="70%" height="70%" align="center" />


Find all possible visualizations with interactive examples in the documentation 
[here](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html). 

## Embedding Models
BERTopic supports many embedding models that can be used to embed the documents and words:
* Sentence-Transformers
* Flair
* Spacy
* Gensim
* USE

[**Sentence-Transformers**](https://github.com/UKPLab/sentence-transformers) is typically used as it has shown great results embedding documents 
meant for semantic similarity. Simply select any from their documentation 
[here](https://www.sbert.net/docs/pretrained_models.html) and pass it to BERTopic:

```python
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
```

[**Flair**](https://github.com/flairNLP/flair) allows you to choose almost any 🤗 transformers model. Simply 
select any from [here](https://huggingface.co/models) and pass it to BERTopic:

```python
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
topic_model = BERTopic(embedding_model=roberta)
```

Click [here](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html) 
for a full overview of all supported embedding models. 

## Dynamic Topic Modeling
Dynamic topic modeling (DTM) is a collection of techniques aimed at analyzing the evolution of topics 
over time. These methods allow you to understand how a topic is represented over time. 
Here, we will be using all of Donald Trump's tweet to see how he talked over certain topics over time: 

```python
import re
import pandas as pd

trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()
```

Then, we need to extract the global topic representations by simply creating and training a BERTopic model:

```python
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(tweets)
```

From these topics, we are going to generate the topic representations at each timestamp for each topic. We do this 
by simply calling `topics_over_time` and pass in his tweets, the corresponding timestamps, and the related topics:

```python
topics_over_time = topic_model.topics_over_time(tweets, topics, timestamps, nr_bins=20)
```

Finally, we can visualize the topics by simply calling `visualize_topics_over_time()`: 

```python
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=6)
```

<img src="images/dtm.gif" width="80%" height="80%" align="center" />

## Overview
BERTopic has quite a number of functions that quickly can become overwhelming. To alleviate this issue, you will find an overview 
of all methods and a short description of its purpose. 

### Common
For quick access to common functions, here is an overview of BERTopic's main methods:

| Method | Code  | 
|-----------------------|---|
| Fit the model    |  `.fit(docs)` |
| Fit the model and predict documents  |  `.fit_transform(docs)` |
| Predict new documents    |  `.transform([new_doc])` |
| Access single topic   | `.get_topic(topic=12)`  |   
| Access all topics     |  `.get_topics()` |
| Get topic freq    |  `.get_topic_freq()` |
| Get all topic information|  `.get_topic_info()` |
| Get representative docs per topic |  `.get_representative_docs()` |
| Update topic representation | `.update_topics(docs, topics, n_gram_range=(1, 3))` |
| Generate topic labels | `.generate_topic_labels()` |
| Set topic labels | `.set_topic_labels(my_custom_labels)` |
| Reduce nr of topics | `.reduce_topics(docs, topics, nr_topics=30)` |
| Find topics | `.find_topics("vehicle")` |
| Save model    |  `.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `.get_params()` |

### Variations
There are many different use cases in which topic modeling can be used. As such, a number of 
variations of BERTopic have been developed such that one package can be used across across many use cases:

| Method | Code  | 
|-----------------------|---|
| (semi-) Supervised Topic Modeling | `.fit(docs, y=y)` |
| Topic Modeling per Class | `.topics_per_class(docs, topics, classes)` |
| Dynamic Topic Modeling | `.topics_over_time(docs, topics, timestamps)` |
| Hierarchical Topic Modeling | `.hierarchical_topics(docs, topics)` |
| Guided Topic Modeling | `BERTopic(seed_topic_list=seed_topic_list)` |

### Visualizations
Evaluating topic models can be rather difficult due to the somewhat subjective nature of evaluation. 
Visualizing different aspects of the topic model helps in understanding the model and makes it easier 
to tweak the model to your liking. 

| Method | Code  | 
|-----------------------|---|
| Visualize Topics    |  `.visualize_topics()` |
| Visualize Documents    |  `.visualize_documents()` |
| Visualize Document Hierarchy    |  `.visualize_hierarchical_documents()` |
| Visualize Topic Hierarchy    |  `.visualize_hierarchy()` |
| Visualize Topic Tree   |  `.get_topic_tree(hierarchical_topics)` |
| Visualize Topic Terms    |  `.visualize_barchart()` |
| Visualize Topic Similarity  |  `.visualize_heatmap()` |
| Visualize Term Score Decline  |  `.visualize_term_rank()` |
| Visualize Topic Probability Distribution    |  `.visualize_distribution(probs[0])` |
| Visualize Topics over Time   |  `.visualize_topics_over_time(topics_over_time)` |
| Visualize Topics per Class | `.visualize_topics_per_class(topics_per_class)` | 

## Citation
To cite the [BERTopic paper](https://arxiv.org/abs/2203.05794), please use the following bibtex reference:

```bibtext
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```
