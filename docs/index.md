# **BERTopic**

<img src="logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. It even supports 
visualizations similar to LDAvis! 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99) 
and [here](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8?sk=03c2168e9e74b6bda2a1f3ed953427e4).

## **Installation**

Installation, with sentence-transformers, can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

You may want to install more depending on the transformers and language backends that you will be using. 
The possible installations are: 

To use Flair embeddings, install BERTopic as follows:
```bash
pip install bertopic[flair]
pip install bertopic[gensim]
pip install bertopic[spacy]
pip install bertopic[use]
```

To install all backends:

```bash
pip install bertopic[all]
```

## **Quick Start**
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
>>> topic_model.get_topic_info()

Topic	Count	Name
-1	    4630	-1_can_your_will_any
49	    693	    49_windows_drive_dos_file
32	    466	    32_jesus_bible_christian_faith
2	    441	    2_space_launch_orbit_lunar
22	    381	    22_key_encryption_keys_encrypted
```

-1 refers to all outliers and should typically be ignored. Next, let's take a look at the most 
frequent topic that was generated, topic 49:

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


## **Overview**
For quick access to common function, here is an overview of BERTopic's main methods:

| Methods | Code  | 
|-----------------------|---|
| Fit the model    |  `BERTopic().fit(docs)` |
| Fit the model and predict documents    |  `BERTopic().fit_transform(docs)` |
| Predict new documents    |  `BERTopic().transform([new_doc])` |
| Access single topic   | `BERTopic().get_topic(topic=12)`  |   
| Access all topics     |  `BERTopic().get_topics()` |
| Get topic freq    |  `BERTopic().get_topic_freq()` |
| Get all topic information|  `BERTopic().get_topic_info()` |
| Dynamic Topic Modeling | `BERTopic().topics_over_time(docs, topics, timestamps)` |
| Visualize Topics    |  `BERTopic().visualize_topics()` |
| Visualize Topic Probability Distribution    |  `BERTopic().visualize_distribution(probabilities[0])` |
| Visualize Topics over Time   |  `BERTopic().visualize_topics_over_time(topics_over_time)` |
| Update topic representation | `BERTopic().update_topics(docs, topics, n_gram_range=(1, 3))` |
| Reduce nr of topics | `BERTopic().reduce_topics(docs, topics, nr_topics=30)` |
| Find topics | `BERTopic().find_topics("vehicle")` |
| Save model    |  `BERTopic().save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `BERTopic().get_params()` |
 
## **Citation**
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