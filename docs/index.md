# **BERTopic**

<img src="logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. It even supports 
visualizations similar to LDAvis! 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99) 
and [here](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8?sk=03c2168e9e74b6bda2a1f3ed953427e4).

###  **Installation**

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

To use Flair embeddings, install BERTopic as follows:
```bash
pip install bertopic[flair]
```

###  **Usage**

Below is an example of how to use the model. The example uses the 
[20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset.  

You can also follow along with the Google Colab notebook [here](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing).

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

### Overview

| Methods | Code  | 
|-----------------------|---|
| Fit the model    |  `topic_model.fit(docs])` |
| Fit the model and predict documents    |  `topic_model.fit_transform(docs])` |
| Predict new documents    |  `topic_model.transform([new_doc])` |
| Access single topic   | `topic_model.get_topic(12)`  |   
| Access all topics     |  `topic_model.get_topics()` |
| Get topic freq    |  `topic_model.get_topic_freq()` |
| Visualize Topics    |  `topic_model.visualize_topics()` |
| Visualize Topic Probability Distribution    |  `topic_model.visualize_distribution(probabilities[0])` |
| Update topic representation | `topic_model.update_topics(docs, topics, n_gram_range=(1, 3))` |
| Reduce nr of topics | `topic_model.reduce_topics(docs, topics, nr_topics=30)` |
| Find topics | `topic_model.find_topics("vehicle")` |
| Save model    |  `topic_model.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `topic_model.get_params()` |
 
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