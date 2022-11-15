BERTopic approaches topic modeling as a cluster task and attempts to cluster semantically similar documents in order to extract common topics. A disadvantage of using such a method is that each document is assigned to a single cluster and therefore also a single topic. In practice, documents may contain a mixture of topics. This can be accounted for by splitting up the documents into sentences and feeding those to BERTopic. Another option is to use a cluster model that can perform soft-clustering, like HDBSCAN. As BERTopic focuses on modularity, we may still want to model that mixture of topics even when we are using a hard-clustering model, like k-Means without the need to split up our documents. This is where `.approximate_distribution` comes in!

In order to perform this approximation, each document is split into tokens according to the provided tokenizer in the `CountVectorizer`. Then, a sliding window is applied on each document creating subsets of the document. For example, with a window size of 3 and stride of 1, the document: 
    
    `Solving the right problem is difficult.`
    
can be split up into `solving the right`, `the right problem`, `right problem is`, and `problem is difficult`. These are called tokensets. 
For each of these tokensets, we calculate their c-TF-IDF representation and find out how similar they are to the previously generated topics. 
Then, the similarities to the topics for each tokenset are summed in order to create a topic distribution for the entire document. 
Although it is often said that documents can contain a mixture of topics, these are often modeled by assigning each word to a single topic. 
With this approach, we take into account that there may be multiple topics for a single word. 

We can make this multiple-topic word assignment a bit more accurate by then splitting these tokensets up into individual tokens and assigning
the topic distributions for each tokenset to each individual token. That way, we can visualize the extent to which a certain word contributes 
to a document's topic distribution.

## **Example**


```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Train our model
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)

# Calculate topic distribution
topic_distr, _ = topic_model.approximate_distribution(docs)
```