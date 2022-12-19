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
[**supervised**](https://maartengr.github.io/BERTopic/getting_started/supervised/supervised.html), 
[**semi-supervised**](https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html), 
[**manual**](https://maartengr.github.io/BERTopic/getting_started/manual/manual.html), 
[**long-document**](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html),
[**hierarchical**](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html), 
[**class-based**](https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html),
[**dynamic**](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html), and 
[**online**](https://maartengr.github.io/BERTopic/getting_started/online/online.html) topic modeling. It even supports visualizations similar to LDAvis!

Corresponding medium posts can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99), [here](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8?sk=03c2168e9e74b6bda2a1f3ed953427e4) and [here](https://towardsdatascience.com/using-whisper-and-bertopic-to-model-kurzgesagts-videos-7d8a63139bdf?sk=b1e0fd46f70cb15e8422b4794a81161d). For a more detailed overview, you can read the [paper](https://arxiv.org/abs/2203.05794) or see a [brief overview](https://maartengr.github.io/BERTopic/algorithm/algorithm.html). 

## Installation

Installation, with sentence-transformers, can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

If you want to install BERTopic with other embedding models, you can choose one of the following:

```bash
pip install bertopic[flair]
pip install bertopic[gensim]
pip install bertopic[spacy]
pip install bertopic[use]
```

## Getting Started
For an in-depth overview of the features of BERTopic 
you can check the [**full documentation**](https://maartengr.github.io/BERTopic/) or you can follow along 
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

The `-1` topic refers to all outlier documents and are typically ignored. Next, let's take a look at the most 
frequent topic that was generated:

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

Using `.get_document_info`, we can also extract information on a document level, such as their corresponding topics, probabilities, whether they are representative documents for a topic, etc.:

```python
>>> topic_model.get_document_info(docs)

Document                               Topic	Name	                        Top_n_words                     Probability    ...
I am sure some bashers of Pens...	0	0_game_team_games_season	game - team - games...	        0.200010       ...
My brother is in the market for...      -1     -1_can_your_will_any	        can - your - will...	        0.420668       ...
Finally you said what you dream...	-1     -1_can_your_will_any	        can - your - will...            0.807259       ...
Think! It's the SCSI card doing...	49     49_windows_drive_dos_file	windows - drive - docs...	0.071746       ...
1) I have an old Jasmine drive...	49     49_windows_drive_dos_file	windows - drive - docs...	0.038983       ...
```

> **Note**
>
> Use `BERTopic(language="multilingual")` to select a model that supports 50+ languages. 

## Visualize Topics
After having trained our BERTopic model, we can iteratively go through hundreds of topics to get a good 
understanding of the topics that were extracted. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
topic_model.visualize_topics()
``` 

<img src="images/topic_visualization.gif" width="60%" height="60%" align="center" />

Find all possible visualizations with interactive examples in the documentation 
[here](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html). 


## Modularity
By default, the main steps for topic modeling with BERTopic are sentence-transformers, UMAP, HDBSCAN, and c-TF-IDF run in sequence. However, it assumes some independence between these steps which makes BERTopic quite modular. In other words, BERTopic not only allows you to build your own topic model but to explore several topic modeling techniques on top of your customized topic model:

https://user-images.githubusercontent.com/25746895/205490350-cd9833e7-9cd5-44fa-8752-407d748de633.mp4

You can swap out any of these models or even remove them entirely. Starting with the embedding step, you can find out how to do this [here](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html) and more about the underlying algorithm and assumptions [here](https://maartengr.github.io/BERTopic/algorithm/algorithm.html). 

## Functionality
BERTopic has many functions that quickly can become overwhelming. To alleviate this issue, you will find an overview 
of all methods and a short description of its purpose. 

### Common
Below, you will find an overview of common functions in BERTopic. 

| Method | Code  | 
|-----------------------|---|
| Fit the model    |  `.fit(docs)` |
| Fit the model and predict documents  |  `.fit_transform(docs)` |
| Predict new documents    |  `.transform([new_doc])` |
| Access single topic   | `.get_topic(topic=12)`  |   
| Access all topics     |  `.get_topics()` |
| Get topic freq    |  `.get_topic_freq()` |
| Get all topic information|  `.get_topic_info()` |
| Get all document information|  `.get_document_info(docs)` |
| Get representative docs per topic |  `.get_representative_docs()` |
| Update topic representation | `.update_topics(docs, n_gram_range=(1, 3))` |
| Generate topic labels | `.generate_topic_labels()` |
| Set topic labels | `.set_topic_labels(my_custom_labels)` |
| Merge topics | `.merge_topics(docs, topics_to_merge)` |
| Reduce nr of topics | `.reduce_topics(docs, nr_topics=30)` |
| Reduce outliers | `.reduce_outliers(docs, topics)` |
| Find topics | `.find_topics("vehicle")` |
| Save model    |  `.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `.get_params()` |


### Attributes
After having trained your BERTopic model, several attributes are saved within your model. These attributes, in part, 
refer to how model information is stored on an estimator during fitting. The attributes that you see below all end in `_` and are 
public attributes that can be used to access model information. 

| Attribute | Description |
|------------------------|---------------------------------------------------------------------------------------------|
| `.topics_`               | The topics that are generated for each document after training or updating the topic model. |
| `.probabilities_` | The probabilities that are generated for each document if HDBSCAN is used. |
| `.topic_sizes_`           | The size of each topic                                                                      |
| `.topic_mapper_`          | A class for tracking topics and their mappings anytime they are merged/reduced.             |
| `.topic_representations_` | The top *n* terms per topic and their respective c-TF-IDF values.                             |
| `.c_tf_idf_`              | The topic-term matrix as calculated through c-TF-IDF.                                       |
| `.topic_labels_`          | The default labels for each topic.                                                          |
| `.custom_labels_`         | Custom labels for each topic as generated through `.set_topic_labels`.                                                               |
| `.topic_embeddings_`      | The embeddings for each topic if `embedding_model` was used.                                                              |
| `.representative_docs_`   | The representative documents for each topic if HDBSCAN is used.                                                |


### Variations
There are many different use cases in which topic modeling can be used. As such, several variations of BERTopic have been developed such that one package can be used across many use cases.

| Method | Code  | 
|-----------------------|---|
| [Topic Distribution Approximation](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html) | `.approximate_distribution(docs)` |
| [Online Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/online/online.html) | `.partial_fit(doc)` |
| [Semi-supervised Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html) | `.fit(docs, y=y)` |
| [Supervised Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/supervised/supervised.html) | `.fit(docs, y=y)` |
| [Manual Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/manual/manual.html) | `.fit(docs, y=y)` |
| [Topic Modeling per Class](https://maartengr.github.io/BERTopic/getting_started/topicsperclass/topicsperclass.html) | `.topics_per_class(docs, classes)` |
| [Dynamic Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html) | `.topics_over_time(docs, timestamps)` |
| [Hierarchical Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) | `.hierarchical_topics(docs)` |
| [Guided Topic Modeling](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html) | `BERTopic(seed_topic_list=seed_topic_list)` |


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
