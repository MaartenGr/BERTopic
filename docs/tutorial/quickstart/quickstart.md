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

To install all backends:

```bash
pip install bertopic[all]
```

## **Quick Start**
We start by extracting topics from the well-known 20 newsgroups dataset which is comprised of English documents:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)
```

After generating topics, we can access the frequent topics that were generated:

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

## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
understanding of the topics that were extracted. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
topic_model.visualize_topics()
``` 

<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

## **Save/Load BERTopic model**
We can easily save a trained BERTopic model by calling `save`:
```python
from bertopic import BERTopic
topic_model = BERTopic()
topic_model.save("my_model")
```

Then, we can load the model in one line:
```python
topic_model = BERTopic.load("my_model")
```

If you do not want to save the embedding model because it is loaded from the cloud, simply run 
`model.save("my_model", save_embedding_model=False)` instead. Then, you can load in the model 
with `BERTopic.load("my_model", embedding_model="whatever_model_you_used")`. 