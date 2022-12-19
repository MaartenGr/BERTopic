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
We start by extracting topics from the well-known 20 newsgroups dataset which is comprised of English documents:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
```

After generating topics, we can access the frequent topics that were generated:

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

Using `.get_document_info`, we can also extract information on a document level, such as their corresponding topics, probabilities, whether they are representative documents for a topic, etc.:

```python
>>> topic_model.get_document_info(docs)

Document                               Topic	Name	                    Top_n_words                     Probability    ...
I am sure some bashers of Pens...	    0	    0_game_team_games_season	game - team - games...	        0.200010       ...
My brother is in the market for...      -1     -1_can_your_will_any	        can - your - will...	        0.420668       ...
Finally you said what you dream...	    -1     -1_can_your_will_any	        can - your - will...            0.807259       ...
Think! It is the SCSI card doing...	    49     49_windows_drive_dos_file	windows - drive - docs...	    0.071746       ...
1) I have an old Jasmine drive...	    49     49_windows_drive_dos_file	windows - drive - docs...	    0.038983       ...
```

!!! Tip "Tip!"
    Use `BERTopic(language="multilingual")` to select a model that supports 50+ languages. 

## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topics to get a good 
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

!!! Tip "Tip!"
    If you do not want to save the embedding model because it is loaded from the cloud, simply run 
    `model.save("my_model", save_embedding_model=False)` instead. Then, you can load in the model 
    with `BERTopic.load("my_model", embedding_model="whatever_model_you_used")`. 

!!! Warning "Warning"
    When saving the model, make sure to also keep track of the versions of dependencies and Python used. 
    Loading and saving the model should be done using the same dependencies and Python. Moreover, models 
    saved in one version of BERTopic should not be loaded in other versions. 
