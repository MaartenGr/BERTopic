BERTopic uses HDBSCAN for clustering the data and it cannot specify the number of clusters you would want. To a certain extent, 
this is an advantage, as we can trust HDBSCAN to be better in finding the number of clusters than we are.
Instead, we can try to reduce the number of topics that have been created. Below, you will find three methods of doing 
so. 
  
### **Manual Topic Reduction**
Each resulting topic has its feature vector constructed from c-TF-IDF. Using those feature vectors, we can find the most similar 
topics and merge them. If we do this iteratively, starting from the least frequent topic, we can reduce the number of topics quite easily. We do this until we reach the value of `nr_topics`:  

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics=20)
```

It is also possible to manually select certain topics that you believe should be merged. 
For example, if topic 1 is `1_space_launch_moon_nasa` and topic 2 is `2_spacecraft_solar_space_orbit` 
it might make sense to merge those two topics: 

```python
topics_to_merge = [1, 2]
topic_model.merge_topics(docs, topics_to_merge)
```

If you have several groups of topics you want to merge, create a list of lists instead:

```python
topics_to_merge = [[1, 2]
                   [3, 4]]
topic_model.merge_topics(docs, topics_to_merge)
```

### **Automatic Topic Reduction**
One issue with the approach above is that it will merge topics regardless of whether they are very similar. They 
are simply the most similar out of all options. This can be resolved by reducing the number of topics automatically. 
To do this, we can use HDBSCAN to cluster our topics using each c-TF-IDF representation. Then, we merge topics that are clustered together. 
Another benefit of HDBSCAN is that it generates outliers. These outliers prevent topics from being merged if no other topics are similar.

To use this option, we simply set `nr_topics` to `"auto"`:

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics="auto")
```

### **Topic Reduction after Training**
Finally, we can also reduce the number of topics after having trained a BERTopic model. The advantage of doing so is that you can decide the number of topics after knowing how many are created. It is difficult to predict before training your model how many topics that are in your documents and how many will be extracted. 
Instead, we can decide afterward how many topics seem realistic:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
# Create topics -> Typically over 50 topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

# Further reduce topics
topic_model.reduce_topics(docs, nr_topics=30)

# Access updated topics
topics = topic_model.topics_
```

The reasoning for putting `docs` as a parameter is that the documents are not saved within 
BERTopic on purpose. If you were to have a million documents, it is very inefficient to save those in BERTopic instead of a dedicated database.  

