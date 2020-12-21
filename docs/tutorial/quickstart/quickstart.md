## **Installation**

**[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher is recommended. If the install below gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

## **Quick Start**
Below is an example of how to use the model. The example uses the 
[20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset.  

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

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

**NOTE**: If you get less than 10 topics, it is advised to decrease the `min_topic_size` in `BERTopic`. This 
will allow clusters to be created more easily and will typically result in more clusters.   


### **Languages**
BERTopic is set to `english` but supports essentially any language for which a document embedding model exists. 
You can choose the language by simply setting the `language` parameter in BERTopic. 

```python
from bertopic import BERTopic
model = BERTopic(language="Dutch")
```

For a list of supported languages, please select the link below. 

<details>
<summary>Supported Languages</summary>

The following languages are supported:
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
</details>  


### **Embedding model**
If you want to select any model from `sentence-transformers` you can simply select that model and pass it through 
BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

Click [here](https://www.sbert.net/docs/pretrained_models.html) for a list of supported sentence transformers models.  


### **Visualize Topic Probabilities**

The variable `probabilities` that is returned from `transform()` or `fit_transform()` can 
be used to understand how confident BERTopic is that certain topics can be found in a document. 

To visualize the distributions, we simply call:
```python
# Make sure to input the probabilities of a single document!
model.visualize_distribution(probabilities[0])
```

<img src="probabilities.png" width="75%" height="75%"/>


**NOTE**: The distribution of the probabilities does not give an indication to 
the distribution of the frequencies of topics across a document. It merely shows
how confident BERTopic is that certain topics can be found in a document.

### **Save/Load BERTopic model**
We can easily save a trained BERTopic model by calling `save`:
```python
from bertopic import BERTopic
model = BERTopic()
model.save("my_model")
```

Then, we can load the model in one line:
```python
loaded_model = BERTopic.load("my_model")
```
