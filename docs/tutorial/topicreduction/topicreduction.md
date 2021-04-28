In `BERTopic`, there are several arguments that might be helpful if you tend to end up with too many or too few 
topics. 

## **Topic Parameters**
The arguments discussed here all relate to the cluster step of BERTopic. 

#### **Minimum topic size**
The `min_topic_size` parameter is actually used in `HDBSCAN`. It tells HDBSCAN what the minimum size of a cluster 
should be before it is accepted as a cluster. When you set this parameter very high, you will get very little clusters 
as they all need to be high. In constrast, if you set this too low you might end with too many extremely specific 
clusters. 

```python
from bertopic import BERTopic
topic_model = BERTopic(min_topic_size=10)
```

You can increase this value if you have more data available or if you expect clusters to be quite large. 

## **Hierarchical Topic Reduction**
It is not possible for HDBSCAN to specify the number of clusters you would want. To a certain extent, 
this is actually an advantage, as we can trust HDBSCAN to be better in finding the number of clusters than we are.
Instead, we can try to reduce the number of topics that have been created. Below, you will find three methods of doing 
so. 
  
### **Manual Topic Reduction**
Each resulting topic has its own 
feature vector constructed from c-TF-IDF. Using those feature vectors, we can find the most similar 
topics and merge them. If we do this iteratively, starting from the least frequent topic, we can reduce the number 
of topics quite easily. We do this until we reach the value of `nr_topics`:  

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics=20)
```

### **Automatic Topic Reduction**
One issue with the approach above is that it will merge topics regardless of whether they are actually very similar. They 
are simply the most similar out of all options. This can be resolved by reducing the number of topics automatically. 
It will reduce the number of topics, starting from the least frequent topic, as long as it exceeds a minimum 
similarity of 0.915. To use this option, we simply set `nr_topics` to `"auto"`:

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics="auto")
```

### **Topic Reduction after Training**
Finally, we can also reduce the number of topics after having trained a BERTopic model. The advantage of doing so, 
is that you can decide the number of topics after knowing how many are actually created. It is difficult to 
predict before training your model how many topics that are in your documents and how many will be extracted. 
Instead, we can decide afterwards how many topics seems realistic:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
# Create topics -> Typically over 50 topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic(calculate_probabilities=True)
topics, probs = topic_model.fit_transform(docs)

# Further reduce topics
new_topics, new_probs = topic_model.reduce_topics(docs, topics, probabilities=probs, nr_topics=30)
```

The reasoning for putting `docs` and `topics` (and optionally `probabilities`) as parameters is that these values are not saved within 
BERTopic on purpose. If you were to have a million documents, it is very inefficient to save those in BERTopic 
instead of a dedicated database.  

