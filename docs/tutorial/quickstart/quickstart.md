## **Installation**

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```bash
pip install bertopic
```

To use the visualization options, install BERTopic as follows:

```bash
pip install bertopic[visualization]
```

To use Flair embeddings, install BERTopic as follows:
```bash
pip install bertopic[flair]
```

To install all additional dependencies:

```bash
pip install bertopic[all]
```

## **Quick Start**
We start by extracting topics from the well-known 20 newsgroups dataset which is comprised of english documents:


```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)
```

After generating topics and their probabilities, we can access the frequent topics that were generated:

```python
>>> topic_model.get_topic_freq().head()
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
>>> topic_model.get_topic(49)
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

**NOTE**: Use `BERTopic(language="multilingual")` to select a model that supports 50+ languages.

## **Visualize Topics**
After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
understanding of the topics that were extract. However, that takes quite some time and lacks a global representation. 
Instead, we can visualize the topics that were generated in a way very similar to 
[LDAvis](https://github.com/cpsievert/LDAvis):

```python
topic_model.visualize_topics()
``` 

<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>

## **Embedding Models**
The parameter `embedding_model` takes in a string pointing to a sentence-transformers model, 
a SentenceTransformer, or a Flair DocumentEmbedding model. 

### **Sentence-Transformers**  
You can select any model from `sentence-transformers` [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

Or select a SentenceTransformer model with your own parameters:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cpu")
topic_model = BERTopic(embedding_model=sentence_model)
```

### **Flair**
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
topic_model = BERTopic(embedding_model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).

### **Custom Embeddings**    
You can also use previously generated embeddings by passing it through `fit_transform()`:

```python
from bertopic import BERTopic
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs, embeddings)
```

## **Save/Load BERTopic model**
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

If you do not want to save the embedding model because it is loaded from the cloud, simply run 
`model.save("my_model", save_embedding_model=False)` instead. Then, you can load in the model 
with `BERTopic.load("my_model", embedding_model="whatever_model_you_used")`. 