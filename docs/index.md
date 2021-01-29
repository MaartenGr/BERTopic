# **BERTopic**

<img src="logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. It even supports 
visualizations similar to LDAvis! 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99).

###  **Installation**

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

To use the visualization options, install BERTopic as follows:

```bash
pip install bertopic[visualization]
```

###  **Usage**

Below is an example of how to use the model. The example uses the 
[20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset.  

You can also follow along with the Google Colab notebook [here](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing).

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

model = BERTopic()
topics, probabilities = model.fit_transform(docs)
```

After generating topics and their probabilities, we can access the frequent topics that were generated:

```python
>>> model.get_topic_freq().head()
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
>>> model.get_topic(49)
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

<details>
<summary>Supported Languages</summary>

<br>
Use <b>"multilingual"</b> to select a model that supports 50+ languages. 
<br><br>
Moreover, the following <b>languages</b> are supported: <br>
Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese,
Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanize, Bosnian,
Breton, Bulgarian, Burmese, Burmese zawgyi font, Catalan, Chinese (Simplified),
Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto,
Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek,
Gujarati, Hausa, Hebrew, Hindi, Hindi Romanize, Hungarian, Icelandic, Indonesian,
Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean,
Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian,
Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian,
Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian,
Russian, Sanskrit, Scottish Gaelic, Serbian, Sindhi, Sinhala, Slovak,
Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil,
Tamil Romanize, Telugu, Telugu Romanize, Thai, Turkish, Ukrainian,
Urdu, Urdu Romanize, Uyghur, Uzbek, Vietnamese, Welsh, Western Frisian,
Xhosa, Yiddish
<br>
</details>  


### Overview

| Methods | Code  | 
|-----------------------|---|
| Fit the model    |  `model.fit(docs])` |
| Fit the model and predict documents    |  `model.fit_transform(docs])` |
| Predict new documents    |  `model.transform([new_doc])` |
| Access single topic   | `model.get_topic(12)`  |   
| Access all topics     |  `model.get_topics()` |
| Get topic freq    |  `model.get_topic_freq()` |
| Visualize Topics    |  `model.visualize_topics()` |
| Visualize Topic Probability Distribution    |  `model.visualize_distribution(probabilities[0])` |
| Update topic representation | `model.update_topics(docs, topics, n_gram_range=(1, 3))` |
| Reduce nr of topics | `model.reduce_topics(docs, topics, probabilities, nr_topics=30)` |
| Find topics | `model.find_topics("vehicle")` |
| Save model    |  `model.save("my_model")` |
| Load model    |  `BERTopic.load("my_model")` |
| Get parameters |  `model.get_params()` |
