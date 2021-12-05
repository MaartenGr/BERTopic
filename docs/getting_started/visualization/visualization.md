Visualizing BERTopic and its derivatives is important in understanding the model, how it works, but more importantly, where it works. 
Since topic modeling can be quite a subjective field it is difficult for users to validate their models. Looking at the topics and seeing 
if they make sense is an important factor in eliviating this issue. 

## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through hundreds of topics to get a good 
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
topics, probs = topic_model.fit_transform(docs)
```

Then, we simply call `topic_model.visualize_topics()` in order to visualize our topics. The resulting graph is a 
plotly interactive graph which can be converted to HTML. 

Thus, you can play around with the results below:

<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

You can use the slider to select the topic which then lights up red. If you hover over a topic, then general 
information is given about the topic, including the size of the topic and its corresponding words.


## **Visualize Topic Hierarchy**
The topics that were created can be hierarchically reduced. In order to understand the potential hierarchical 
structure of the topics, we can use `scipy.cluster.hierarchy` to create clusters and visualize how 
they relate to one another. This might help selecting an appropriate `nr_topics` when reducing the number 
of topics that you have created. To visualize this hierarchy, simply call `topic_model.visualize_hierarchy()`:

<iframe src="hierarchy.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

!!! note
    Do note that this is not the actual procedure of `.reduce_topics()` when `nr_topics` is set to 
    auto since HDBSCAN is used to automatically extract topics. The visualization above closely resembles 
    the actual procedure of `.reduce_topics()` when any number of `nr_topics` is selected. 

## **Visualize Terms**
We can visualize the selected terms for a few topics by creating bar charts out of the c-TF-IDF scores 
for each topic representation. Insights can be gained from the relative c-TF-IDF scores between and within 
topics. Moreover, you can easily compare topic representations to each other. 
To visualize this hierarchy, simply call `topic_model.visualize_barchart()`:

<iframe src="bar_chart.html" style="width:1100px; height: 660px; border: 0px;""></iframe>


## **Visualize Topic Similarity**
Having generated topic embeddings, through both c-TF-IDF and embeddings, we can create a similarity 
matrix by simply applying cosine similarities through those topic embeddings. The result will be a 
matrix indicating how similar certain topics are to each other. 
To visualize the heatmap, simply call `topic_model.visualize_heatmap()`:
 
<iframe src="heatmap.html" style="width:1000px; height: 720px; border: 0px;""></iframe>

Note that you can set `n_clusters` in `visualize_heatmap` to order the topics by their similarity. 
This will result in blocks being formed in the heatmap indicating which clusters of topics are 
similar to each other. This step is very much recommended as it will make reading the heatmap easier.      


## **Visualize Term Score Decline**
Topics are represented by a number of words starting with the best representative word. 
Each word is represented by a c-TF-IDF score. The higher the score, the more representative a word 
to the topic is. Since the topic words are sorted by their c-TF-IDF score, the scores slowly decline 
with each word that is added. At some point adding words to the topic representation only marginally 
increases the total c-TF-IDF score and would not be beneficial for its representation. 

To visualize this effect, we can plot the c-TF-IDF scores for each topic by the term rank of each word. 
In other words, the position of the words (term rank), where the words with 
the highest c-TF-IDF score will have a rank of 1, will be put on the x-axis. Whereas the y-axis 
will be populated by the c-TF-IDF scores. The result is a visualization that shows you the decline 
of c-TF-IDF score when adding words to the topic representation. It allows you, using the elbow method, 
the select the best number of words in a topic. 

To visualize the c-TF-IDF score decline, simply call `topic_model.visualize_term_rank()`:

<iframe src="term_rank.html" style="width:1000px; height: 530px; border: 0px;""></iframe>

To enable the log scale on the y-axis for a better view of individual topics, 
simply call `topic_model.visualize_term_rank(log_scale=True)`:

<iframe src="term_rank_log.html" style="width:1000px; height: 530px; border: 0px;""></iframe>

This visualization was heavily inspired by the "Term Probability Decline" visualization found in an 
analysis by the amazing [tmtoolkit](https://tmtoolkit.readthedocs.io/).
Reference to that specific analysis can be found 
[here](https://wzbsocialsciencecenter.github.io/tm_corona/tm_analysis.html). 

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
topics, probs = model.fit_transform(tweets)
topics_over_time = model.topics_over_time(tweets, topics, timestamps)
```

Then, we visualize some interesting topics: 

```python
model.visualize_topics_over_time(topics_over_time, topics=[9, 10, 72, 83, 87, 91])
```
<iframe src="trump.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

## **Visualize Topics per Class**
You might want to extract and visualize the topic representation per class. For example, if you have 
specific groups of users that might approach topics differently, then extracting them would help understanding 
how these users talk about certain topics. In other words, this is simply creating a topic representation for 
certain classes that you might have in your data. 

First, we need to train our model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Prepare data and classes
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
classes = [data["target_names"][i] for i in data["target"]]

# Create topic model and calculate topics per class
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
topics_per_class = topic_model.topics_per_class(docs, topics, classes=classes)
```

Then, we visualize the topic representation of major topics per class: 

```python
topic_model.visualize_topics_per_class(topics_per_class)
```

<iframe src="topics_per_class.html" style="width:1400px; height: 1000px; border: 0px;""></iframe>


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

<iframe src="probabilities.html" style="width:1000px; height: 500px; border: 0px;""></iframe>


!!! note
    The distribution of the probabilities does not give an indication to 
    the distribution of the frequencies of topics across a document. It merely shows
    how confident BERTopic is that certain topics can be found in a document.

