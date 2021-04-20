## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
understanding of the topics that were extract. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis). 

We embed our c-TF-IDF representation of the topics in 2D using Umap and then visualize the two dimensions using 
plotly such that we can create an interactive view.

First, we need to train our model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)
```

Then, we simply call `topic_model.visualize_topics()` in order to visualize our topics. The resulting graph is a 
plotly interactive graph which can be converted to html. 

Thus, you can play around with the results below:

<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

You can use the slider to select the topic which then lights up red. If you hover over a topic, then general 
information is given about the topic, including size of the topic and its corresponding words.

## **Visualize Topics over Time**
After creating topics over time with Dynamic Topic Modeling, we can visualize these topics by 
leveraging the interactive abilities of Plotly. Plotly allows us to show the frequency 
of topics over time whilst giving the option of hovering over the points to show the time-specific topic representations. 
Simply call `visualize_topics_over_time` with the newly created topics over time:


```python
import re
import pandas as pd
from bertopic import BERTopic

# Prepare data
trump = pd.read_csv('https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6')
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()

# Create topics over time
model = BERTopic(verbose=True)
topics, _ = model.fit_transform(tweets)
topics_over_time = model.topics_over_time(tweets, topics, timestamps)
```

Then, we visualize some interesting topics: 

```python
model.visualize_topics_over_time(topics_over_time, topcs=[9, 10, 72, 83, 87, 91])
```
<iframe src="trump.html" style="width:1000px; height: 680px; border: 0px;""></iframe>


## **Visualize Probablities**
We can also calculate the probabilities of topics found in a document. In order to do so, we have to 
set `calculate_probabilities` to True as calculating them can be quite computationally expensive. 
Then, we use the variable `probabilities` that is returned from `transform()` or `fit_transform()` 
to understand how confident BERTopic is that certain topics can be found in a document:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic(calculate_probabilities=True)
topics, probabilities = topic_model.fit_transform(docs)
```

To visualize the distributions, we simply call:

```python
topic_model.visualize_distribution(probabilities[0])
```

<iframe src="probabilities.html" style="width:1000px; height: 680px; border: 0px;""></iframe>


**NOTE**: The distribution of the probabilities does not give an indication to 
the distribution of the frequencies of topics across a document. It merely shows
how confident BERTopic is that certain topics can be found in a document.

