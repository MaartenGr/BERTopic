# **BERTopic**

<img src="logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages BERT embeddings and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99).

###  **Installation**
**[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher is recommended. If the install below gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

###  **Usage**

Below is an example of how to use the model. The example uses the 
[20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset.  

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all')['data']

model = BERTopic()
topics, probabilities = model.fit_transform(docs)
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

###  **Function Overview**

| Methods | Code  | Returns  |
|-----------------------|---|---|
| Access single topic   | `model.get_topic(12)`  | Tuple[Word, Score]  |   
| Access all topics     |  `model.get_topics()` | List[Tuple[Word, Score]]  |
| Get single topic freq |  `model.get_topic_freq(12)` | int |
| Get all topic freq    |  `model.get_topics_freq()` | DataFrame  |
| Fit the model    |  `model.fit(docs])` | BERTopic  |
| Fit the model and predict documents    |  `model.fit_transform(docs])` | List[int], List[float]  |
| Predict new documents    |  `model.transform([new_doc])` | List[int], List[float]  |
| Visualize Topic Probability Distribution    |  `model.visualize_distribution(probabilities)` | Matplotlib.Figure  |
| Save model    |  `model.save("my_model")` | -  |
| Load model    |  `BERTopic.load("my_model")` | - |


## **Google Colaboratory**  
Since we are using transformer-based embeddings you might want to leverage gpu-acceleration
to speed up the model. For that, I have created a tutorial 
[Google Colab Notebook](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)
that you can use to run the model as shown above. 

If you want to tweak the inner workings or follow along with the medium post, use [this](https://colab.research.google.com/drive/1-SOw0WHZ_ZXfNE36KUe3Z-UpAO3vdhGg?usp=sharing)
 notebook instead. 
