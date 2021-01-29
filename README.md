[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/bertopic/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/VLAC/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/BERTopic)](https://pypi.org/project/bertopic/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/BERTopic/Code%20Checks/master)](https://pypi.org/project/bertopic/)
[![docs](https://img.shields.io/badge/docs-Passing-green.svg)](https://maartengr.github.io/BERTopic/)
[![DOI](https://zenodo.org/badge/297672263.svg)](https://zenodo.org/badge/latestdoi/297672263)


# BERTopic

<img src="images/logo.png" width="35%" height="35%" align="right" />

BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters
allowing for easily interpretable topics whilst keeping important words in the topic descriptions. It even supports 
visualizations similar to LDAvis! 

Corresponding medium post can be found [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6?source=friends_link&sk=0b5a470c006d1842ad4c8a3057063a99).

## Installation

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

To use the visualization options, install BERTopic as follows:

```bash
pip install bertopic[visualization]
```

<details>
<summary>Installation Errors</summary>

PyTorch 1.4.0 or higher is recommended. If the install gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 
</details>  

## Getting Started
For an in-depth overview of the features of `BERTopic` 
you can check the full documentation [here](https://maartengr.github.io/BERTopic/) or you can follow along 
with the Google Colab notebook [here](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing).

### Quick Start
We start by extracting topics from the well-known 20 newsgroups dataset which is comprised of english documents:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

model = BERTopic(language="english")
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

### Visualize Topics
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
understanding of the topics that were extract. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
model.visualize_topics()
``` 

<img src="images/topic_visualization.gif" width="60%" height="60%" align="center" />

### Visualize Topic Probabilities

The variable `probabilities` that is returned from `transform()` or `fit_transform()` can 
be used to understand how confident BERTopic is that certain topics can be found in a document. 

To visualize the distributions, we simply call:
```python
# Make sure to input the probabilities of a single document!
model.visualize_distribution(probabilities[0])
```

<img src="images/probabilities.png" width="75%" height="75%"/>

### Embedding Models
You can select any model from `sentence-transformers` and pass it through 
BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

You can also use previously generated embeddings by passing it through `fit_transform()`:

```python
model = BERTopic()
topics, probabilities = model.fit_transform(docs, embeddings)
```

Click [here](https://www.sbert.net/docs/pretrained_models.html) for a list of supported sentence transformers models.  

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
 
### Citation
To cite BERTopic in your work, please use the following bibtex reference:

```bibtex
@misc{grootendorst2020bertopic,
  author       = {Maarten Grootendorst},
  title        = {BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.4.2},
  doi          = {10.5281/zenodo.4430182},
  url          = {https://doi.org/10.5281/zenodo.4430182}
}
```
