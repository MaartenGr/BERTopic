## **Visualize Topics**

## **Visualize Probablities**
The variable probabilities that is returned from transform() or fit_transform() can be used to understand how 
confident BERTopic is that certain topics can be found in a document.

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Create topics
docs = fetch_20newsgroups(subset='train')['data']
model = BERTopic()
topics, probs = model.fit_transform(docs)
```

To visualize the distributions, we simply call:

```python
model.visualize_distribution(probabilities[0])
```

<img src="probabilities.png" width="75%" height="75%"/>