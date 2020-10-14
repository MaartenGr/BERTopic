[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/VLAC/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/BERTopic)](https://pypi.org/project/bertopic/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/BERTopic/Code%20Checks/master)](https://pypi.org/project/bertopic/)

<img src="logo.png" width="35%" height="35%" align="right" />

# BERTopic

BERTopic is a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99).

## About the Project  

The initial purpose of this project was to generalize [Top2Vec](https://github.com/ddangelov/Top2Vec) such that it could be 
used with state-of-art pre-trained transformer models. However, this proved difficult due to the different natures 
of Doc2Vec and transformer models. Instead, I decided to come up with a different algorithm that could use BERT 
and 🤗 transformers embeddings. The results is **BERTopic**, an algorithm for generating topics using state-of-the-art embeddings.  
 
###  Installation
**[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher is recommended. If the install below gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

``pip install bertopic``

###  Usage

Below is an example of how to use the model. The example uses the 
[20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset.  

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all')['data']

model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)
topics = model.fit_transform(docs)
```

The resulting topics can be accessed through `model.get_topic(topic)`:

```python
>>> model.get_topic(9)
[('game', 0.005251396890032802),
 ('team', 0.00482651185323754),
 ('hockey', 0.004335032060690186),
 ('players', 0.0034782716706978963),
 ('games', 0.0032873248432630227),
 ('season', 0.003218987432255393),
 ('play', 0.0031855141725669637),
 ('year', 0.002962343114817677),
 ('nhl', 0.0029577648449943144),
 ('baseball', 0.0029245163154193524)]
``` 

You can find an overview of all models currently in BERTopic [here](https://www.sbert.net/docs/pretrained_models.html) and [here](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0). 

###  Overview


| Methods | Code  | Returns  |
|-----------------------|---|---|
| Access single topic   | `model.get_topic(12)`  | Tuple[Word, Score]  |   
| Access all topics     |  `model.get_topic()` | List[Tuple[Word, Score]]  |
| Get single topic freq |  `model.get_topic_freq(12)` | int |
| Get all topic freq    |  `model.get_topics_freq()` | DataFrame  |
| Fit the model    |  `model.fit(docs])` | -  |
| Predict new documents    |  `model.transform([new_doc])` | List[int]  |
| Save model    |  `model.save("my_model")` | -  |
| Load model    |  `BERTopic.load("my_model")` | - |
   
**NOTE**: The embeddings itself are not preserved in the model as they are only vital for creating the clusters. 
Therefore, it is advised to only use `fit` and then `transform` if you are looking to generalize the model to new documents.
For existing documents, it is best to use `fit_transform` directly as it only needs to generate the document
embeddings once.   

## Google Colaboratory  
Since we are using transformer-based embeddings you might want to leverage gpu-acceleration
to speed up the model. For that, I have created a tutorial 
[Google Colab Notebook](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)
that you can use to run the model as shown above. 

If you want to tweak the inner workings or follow along with the medium post, use [this](https://colab.research.google.com/drive/1-SOw0WHZ_ZXfNE36KUe3Z-UpAO3vdhGg?usp=sharing)
 notebook instead. 

## References
Angelov, D. (2020). [Top2Vec: Distributed Representations of Topics.](https://arxiv.org/abs/2008.09470) *arXiv preprint arXiv*:2008.09470.




