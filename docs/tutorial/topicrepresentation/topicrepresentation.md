The topics that are extracted from BERTopic are represented by words. These words are extracted from the documents 
occupying their topics using a class-based TF-IDF. This allows us to extract words that are interesting to a topic but 
less so to another. 

### Update Topic Representation after Training
When you have trained a model and viewed the topics and the words that represent them,
you might not be satisfied with the representation. Perhaps you forgot to remove
stop_words or you want to try out a different n_gram_range. We can use the function `update_topics` to update 
the topic representation with new parameters for `c-TF-IDF`: 

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Create topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic(n_gram_range=(2, 3))
topics, _ = topic_model.fit_transform(docs)
```

From the model created above, one of the most frequent topics is the following:

```python
>>> topic_model.get_topic(31)[:10]
[('clipper chip', 0.007240771542316232),
 ('key escrow', 0.004601603973377443),
 ('law enforcement', 0.004277247929596332),
 ('intercon com', 0.0035961920238955824),
 ('amanda walker', 0.003474856425297157),
 ('serial number', 0.0029876119137150358),
 ('com amanda', 0.002789303096817983),
 ('intercon com amanda', 0.0027386688593327084),
 ('amanda intercon', 0.002585262048515583),
 ('amanda intercon com', 0.002585262048515583)]
```

Although there does seem to be some relation between words, it is difficult, at least for me, to intuitively understand 
what the topic is about. Instead, let's simplify the topic representation by setting `n_gram_range` to (1, 3) to 
also allow for single words.

```python
>>> topic_model.update_topics(docs, topics, n_gram_range=(1, 3))
>>> topic_model.get_topic(31)[:10]
[('encryption', 0.008021846079148017),
 ('clipper', 0.00789642647602742),
 ('chip', 0.00637127942464045),
 ('key', 0.006363124787175884),
 ('escrow', 0.005030980365244285),
 ('clipper chip', 0.0048271268437973395),
 ('keys', 0.0043245812747907545),
 ('crypto', 0.004311198708675516),
 ('intercon', 0.0038772934659295076),
 ('amanda', 0.003516026493904586)]
```

To me, the combination of the words above seem a bit more intuitive than the words we previously had! You can play 
around with `n_gram_range` or use your own custom `sklearn.feature_extraction.text.CountVectorizer` and pass that  
instead: 

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="English", ngram_range=(1, 5))
topic_model.update_topics(docs, topics, vectorizer_model=vectorizer_model)
```