<img src="https://raw.githubusercontent.com/MaartenGr/BERTopic/master/images/logo.png" height="220" />

[![PyPI - Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://pypi.org/project/vlac/)
[![PyPI - Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/VLAC/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/badge/pypi-v0.1.2-EF6C00.svg)](https://pypi.org/project/bertopic/)

BERTopic is a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. 

Corresponding medium post can be found [here](https://medium.com/@maartengrootendorst).

<a name="toc"/></a>
## Table of Contents  
<!--ts-->
   1. [About the Project](#about)  
   2. [Algorithm](#algorithm)  
        2.1. [Sentence Transformer](#sentence)  
        2.2. [UMAP + HDBSCAN](#umap)  
        2.3. [c-TF-IDF](#ctfidf)  
   3. [Getting Started](#gettingstarted)    
        3.1. [Installation](#installation)    
        3.2. [Basic Usage](#usage)   
        3.3. [Overview](#overview)    
   4. [Google Colaboratory](#colab)   
<!--te-->
 
<a name="about"/></a>
## 1. About the Project
[Back to ToC](#toc)  

The initial purpose of this project was to generalize [Top2Vec](https://github.com/ddangelov/Top2Vec) such that it could be 
used with state-of-art pre-trained transformer models. However, this proved difficult due to the different natures 
of Doc2Vec and transformer models. Instead, I decided to come up with a different algorithm that could use BERT 
and 🤗 transformers embeddings. The results is **BERTopic**, an algorithm for generating topics using state-of-the-art embeddings.  
 

<a name="gettingstarted"/></a>
## 2. Getting Started
[Back to ToC](#toc)  

<a name="installation"/></a>
###  2.1. Installation
**[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher is recommended. If the install below gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

``pip install bertopic``

**PyTorch**  
If you get  

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

<a name="usage"/></a>
###  2.2. Usage

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


<a name="overview"/></a>
###  2.3. Overview


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


<a name="algorithm"/></a>
## 3. Algorithm
[Back to ToC](#toc)  
The algorithm contains, roughly, 3 stages:
* Extract document embeddings with **Sentence Transformers**
* Cluster document embeddings to create groups of similar documents with **UMAP** and **HDBSCAN**
* Extract and reduce topics with **c-TF-IDF**


<a name="sentence"/></a>
###  3.1. Sentence Transformer
We start by creating document embeddings from a set of documents using 
[sentence-transformer](https://github.com/UKPLab/sentence-transformers). These models are pre-trained for many 
language and are great for creating either document- or sentence-embeddings. 

If you have long documents, I would advise you to split up your documents into paragraphs or sentences as a BERT-based
model in `sentence-transformer` typically has a token limit. 

<a name="umap"/></a>
###  3.2. UMAP + HDBSCAN
Next, in order to cluster the documents using a clustering algorithm such as HDBSCAN we first need to 
reduce its dimensionality as HDBCAN is prone to the curse of dimensionality.

<p align="center">
<img src="https://github.com/MaartenGr/BERTopic/raw/master/images/clusters.png"/>
</p>

Thus, we first lower dimensionality with UMAP as it preserves local structure well after which we can 
use HDBSCAN to cluster similar documents.  

<a name="ctfidf"/></a>
###  3.3. c-TF-IDF
What we want to know from the clusters that we generated, is what makes one cluster, based on their content, 
different from another? To solve this, we can modify TF-IDF such that it allows for interesting words per topic
instead of per document. 

When you apply TF-IDF as usual on a set of documents, what you are basically doing is comparing the importance of 
words between documents. Now, what if, we instead treat all documents in a single category (e.g., a cluster) 
as a single document and then apply TF-IDF? The result would be importance scores for words within a cluster. 
The more important words are within a cluster, the more it is representative of that topic. In other words, 
if we extract the most important words per cluster, we get descriptions of **topics**! 

<p align="center">
<img src="https://github.com/MaartenGr/BERTopic/raw/master/images/ctfidf.png" height="50"/>
</p>  

Each cluster is converted to a single document instead of a set of documents. 
Then, the frequency of word `t` are extracted for each class `i` and divided by the total number of words `w`. 
This action can now be seen as a form of regularization of frequent words in the class.
Next, the total, unjoined, number of documents `m` is divided by the total frequency of word `t` across all classes `n`.

<a name="colab"/></a>
## 4. Google Colaboratory
[Back to ToC](#toc)  
Since we are using transformer-based embeddings you might want to leverage gpu-acceleration
to speed up the model. For that, I have created a tutorial 
[Google Colab Notebook](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)
that you can use to run the model as shown above. 

If you want to tweak the inner workings or follow along with the medium post, use [this](https://colab.research.google.com/drive/1-SOw0WHZ_ZXfNE36KUe3Z-UpAO3vdhGg?usp=sharing)
 notebook instead. 



