# BERTopic

<img src="logo.png" width="30%" height="30%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions.

BERTopic supports 
[**guided**](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html), 
(semi-) [**supervised**](https://maartengr.github.io/BERTopic/getting_started/supervised/supervised.html), 
and [**dynamic**](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html) topic modeling. It even supports visualizations similar to LDAvis!

Corresponding medium posts can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99) 
and [here](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8?sk=03c2168e9e74b6bda2a1f3ed953427e4).

## **Installation**

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

## **Quick Start**
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
-1	    4630	-1_can_your_will_any
0	    693	    49_windows_drive_dos_file
1	    466	    32_jesus_bible_christian_faith
2	    441	    2_space_launch_orbit_lunar
3	    381	    22_key_encryption_keys_encrypted
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


## **Overview**
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
| Get topics per class | `.topics_per_class(docs, topics, classes)` |
| Dynamic Topic Modeling | `.topics_over_time(docs, topics, timestamps)` |
| Update topic representation | `.update_topics(docs, topics, n_gram_range=(1, 3))` |
| Reduce nr of topics | `.reduce_topics(docs, topics, nr_topics=30)` |
| Find topics | `.find_topics("vehicle")` |
| Save model    |  `.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `.get_params()` |

For an overview of BERTopic's visualization methods:

| Method | Code  | 
|-----------------------|---|
| Visualize Topics    |  `.visualize_topics()` |
| Visualize Topic Hierarchy    |  `.visualize_hierarchy()` |
| Visualize Topic Terms    |  `.visualize_barchart()` |
| Visualize Topic Similarity  |  `.visualize_heatmap()` |
| Visualize Term Score Decline  |  `.visualize_term_rank()` |
| Visualize Topic Probability Distribution    |  `.visualize_distribution(probs[0])` |
| Visualize Topics over Time   |  `.visualize_topics_over_time(topics_over_time)` |
| Visualize Topics per Class | `.visualize_topics_per_class(topics_per_class)` | 

  
## **Citation**
To cite the [BERTopic paper](https://arxiv.org/abs/2203.05794), please use the following bibtex reference:

```bibtext
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```
