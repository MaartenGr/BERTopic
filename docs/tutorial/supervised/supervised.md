In this tutorial, we will be looking at a new feature of BERTopic, namely (semi)-supervised topic modeling! 
This allows us to steer the dimensionality reduction of the embeddings into a space that closely follows any labels you might already have. 
In other words, we use a semi-supervised UMAP instance to reduce the dimensionality of embeddings before clustering the documents 
with HDBSCAN. 

First, let us prepare the data needed for our topic model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
categories = data["target"]
category_names = data["target_names"]
```

We are using the popular 20 Newsgroups dataset which contains roughly 18000 newsgroups posts that each is 
assigned to one of 20 categories. Using this dataset we can try to extract its corresponding topic model whilst 
taking its underlying categories into account. These categories are here the variable `targets`.

Each document can be put into one of the following categories:

```python
>>> category_names

['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc'] 
```
## **Semi-supervised Topic Modeling**
In semi-supervised topic modeling, we only have some labels for our documents. The documents for which we do have labels 
are used to somewhat guide BERTopic to the extraction of topics for those labels. The documents for which we do not have 
labels are assigned a -1. For this example, imagine we only the labels of categories that are related to computers 
and we want to create a topic model using semi-supervised modeling: 

```python
labels_to_add = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x',]
indices = [category_names.index(label) for label in labels_to_add]
y = [label if label in indices else -1 for label in categories]
``` 

The `y` variable contains many -1 values since we do not know the all the categories. 

Next, we use those newly constructed labels to again BERTopic semi-supervised:

```python
topic_model = BERTopic(verbose=True).fit(docs, y=y)
```

And that is it! By defining certain classes for our documents, we can steer the topic modeling towards modeling the 
pre-defined categories. 

## **Supervised Topic Modeling**
In supervised topic modeling, we have labels for all our documents. This can be pre-defined topics or simply documents  
that you feel belong together regardless of their content. BERTopic will nudge the creation of topics towards these categories 
using the pre-defined labels. 

To perform supervised topic modeling, we simply use all categories:

```python
topic_model = BERTopic(verbose=True).fit(docs, y=categories)
```

The topic model will be much more attuned to the categories that were defined previously. However, this does not mean 
that only topics for these categories will be found. BERTopic is likely to find more specific topics in those you 
have already defined. This allows you to discover previously unknown topics!