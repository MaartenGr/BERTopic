## **Version 0.11.0**
*Release date: 11 July, 2022*

**Highlights**:

* Perform [hierarchical topic modeling](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) with `.hierarchical_topics`

```python 
hierarchical_topics = topic_model.hierarchical_topics(docs, topics) 
```

* Visualize [hierarchical topic representations](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#visualizations) with `.visualize_hierarchy`

```python
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
```

* Extract a [text-based hierarchical topic representation](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#visualizations) with `.get_topic_tree`

```python
tree = topic_model.get_topic_tree(hierarchical_topics)
```

* Visualize [2D documents](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-documents) with `.visualize_documents()`

```python
# Use input embeddings
topic_model.visualize_documents(docs, embeddings=embeddings)

# or use 2D reduced embeddings through a method of your own (e.g., PCA, t-SNE, UMAP, etc.)
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
```

* Visualize [2D hierarchical documents](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-hierarchical-documents) with `.visualize_hierarchical_documents()` 

```python
# Run the visualization with the original embeddings
topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)

# Or, if you have reduced the original embeddings already which speed things up quite a bit:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
```

* Create [custom labels](https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#custom-labels) to the topics throughout most visualizations

```python
# Generate topic labels
topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=10, separator=", ")

# Set them internally in BERTopic
topic_model.set_topic_labels(topics_labels)
```

* Manually [merge topics](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html#merge-topics) with `.merge_topics()`

```python
# Merge topics 1, 2, and 3
topics_to_merge = [1, 2, 3]
topic_model.merge_topics(docs, topics, topics_to_merge)

# Merge topics 1 and 2, and separately merge topics 3 and 4
topics_to_merge = [[1, 2], [3, 4]]
topic_model.merge_topics(docs, topics, topics_to_merge)
```

* Added example for finding similar topics between two models in the [tips & tricks](https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html) page
* Add multi-modal example in the [tips & tricks](https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html) page
* Added native [Hugging Face transformers](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html#hugging-face-transformers) support 

**Fixes**: 

* Fix support for k-Means in `.visualize_heatmap` ([#532](https://github.com/MaartenGr/BERTopic/issues/532))
* Fix missing topic 0 in `.visualize_topics` ([#533](https://github.com/MaartenGr/BERTopic/issues/533))
* Fix inconsistencies in `.get_topic_info` ([#572](https://github.com/MaartenGr/BERTopic/issues/572)) and ([#581](https://github.com/MaartenGr/BERTopic/issues/581))
* Add `optimal_ordering` parameter to `.visualize_hierarchy` by [@rafaelvalero](https://github.com/rafaelvalero) in [#390](https://github.com/MaartenGr/BERTopic/pull/390)
* Fix RuntimeError when used as sklearn estimator by [@simonfelding](https://github.com/simonfelding) in [#448](https://github.com/MaartenGr/BERTopic/pull/448)
* Fix typo in visualization documentation by [@dwhdai](https://github.com/dwhdai) in [#475](https://github.com/MaartenGr/BERTopic/pull/475)
* Fix typo in docstrings by [@xwwwwww](https://github.com/xwwwwww) in [#549](https://github.com/MaartenGr/BERTopic/pull/549)
* Support higher Flair versions



## **Version 0.10.0**
*Release date: 30 April, 2022*


**Highlights**: 

* Use any dimensionality reduction technique instead of UMAP:

```python
from bertopic import BERTopic
from sklearn.decomposition import PCA

dim_model = PCA(n_components=5)
topic_model = BERTopic(umap_model=dim_model)
```

* Use any clustering technique instead of HDBSCAN:

```python
from bertopic import BERTopic
from sklearn.cluster import KMeans

cluster_model = KMeans(n_clusters=50)
topic_model = BERTopic(hdbscan_model=cluster_model)
```

**Documentation**: 

* Add a CountVectorizer page with tips and tricks on how to create topic representations that fit your use case
* Added pages on how to use other dimensionality reduction and clustering algorithms
 * Additional instructions on how to reduce outliers in the FAQ:

```python 
import numpy as np
probability_threshold = 0.01
new_topics = [np.argmax(prob) if max(prob) >= probability_threshold else -1 for prob in probs] 
```

**Fixes**: 

* Fixed `None` being returned for probabilities when transforming unseen documents
* Replaced all instances of `arg:` with `Arguments:` for consistency
* Before saving a fitted BERTopic instance, we remove the stopwords in the fitted CountVectorizer model as it can get quite large due to the number of words that end in stopwords if `min_df` is set to a value larger than 1
* Set `"hdbscan>=0.8.28"` to prevent numpy issues
  * Although this was already fixed by the new release of HDBSCAN, it is technically still possible to install 0.8.27 with BERTopic which leads to these numpy issues
* Update gensim dependency to `>=4.0.0` ([#371](https://github.com/MaartenGr/BERTopic/issues/371))
* Fix topic 0 not appearing in visualizations ([#472](https://github.com/MaartenGr/BERTopic/issues/472))
* Fix ([#506](https://github.com/MaartenGr/BERTopic/issues/506))
* Fix ([#429](https://github.com/MaartenGr/BERTopic/issues/429))
* Fix typo in DTM documentation by [@hp0404](https://github.com/hp0404) in [#386](https://github.com/MaartenGr/BERTopic/pull/386)

## **Version 0.9.4**
*Release date: 14 December, 2021*

A number of fixes, documentation updates, and small features:

* Expose diversity parameter
    * Use `BERTopic(diversity=0.1)` to change how diverse the words in a topic representation are (ranges from 0 to 1)
* Improve stability of topic reduction by only computing the cosine similarity within c-TF-IDF and not the topic embeddings
* Added property to c-TF-IDF that all IDF values should be positive ([#351](https://github.com/MaartenGr/BERTopic/issues/351))
* Improve stability of `.visualize_barchart()` and `.visualize_hierarchy()`
* Major [documentation](https://maartengr.github.io/BERTopic/) overhaul (mkdocs, tutorials, FAQ, images, etc. ) ([#330](https://github.com/MaartenGr/BERTopic/issues/330))
* Drop python 3.6 ([#333](https://github.com/MaartenGr/BERTopic/issues/333))
* Relax plotly dependency ([#88](https://github.com/MaartenGr/BERTopic/issues/88))
* Additional logging for `.transform` ([#356](https://github.com/MaartenGr/BERTopic/issues/356))


## **Version 0.9.3**
*Release date:  17 October, 2021*

* Fix [#282](https://github.com/MaartenGr/BERTopic/issues/282)
    * As it turns out the old implementation of topic mapping was still found in the `transform` function
* Fix [#285](https://github.com/MaartenGr/BERTopic/issues/285)
    * Fix getting all representative docs
* Fix [#288](https://github.com/MaartenGr/BERTopic/issues/288)
    * A recent issue with the package `pyyaml` that can be found in Google Colab


## **Version 0.9.2**
*Release date:  12 October, 2021*

A release focused on algorithmic optimization and fixing several issues:

**Highlights**:  
  
* Update the non-multilingual paraphrase-* models to the all-* models due to improved [performance](https://www.sbert.net/docs/pretrained_models.html)
* Reduce necessary RAM in c-TF-IDF top 30 word [extraction](https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix)

**Fixes**:  

* Fix topic mapping
    * When reducing the number of topics, these need to be mapped to the correct input/output which had some issues in the previous version
    * A new class was created as a way to track these mappings regardless of how many times they were executed
    * In other words, you can iteratively reduce the number of topics after training the model without the need to continuously train the model
* Fix typo in embeddings page ([#200](https://github.com/MaartenGr/BERTopic/issues/200)) 
* Fix link in README ([#233](https://github.com/MaartenGr/BERTopic/issues/233))
* Fix documentation `.visualize_term_rank()` ([#253](https://github.com/MaartenGr/BERTopic/issues/253)) 
* Fix getting correct representative docs ([#258](https://github.com/MaartenGr/BERTopic/issues/258))
* Update [memory FAQ](https://maartengr.github.io/BERTopic/faq.html#i-am-facing-memory-issues-help) with [HDBSCAN pr](https://github.com/MaartenGr/BERTopic/issues/151)

## **Version 0.9.1**
*Release date:  1 September, 2021*

A release focused on fixing several issues:

**Fixes**:  

* Fix TypeError when auto-reducing topics ([#210](https://github.com/MaartenGr/BERTopic/issues/210)) 
* Fix mapping representative docs when reducing topics ([#208](https://github.com/MaartenGr/BERTopic/issues/208))
* Fix visualization issues with probabilities ([#205](https://github.com/MaartenGr/BERTopic/issues/205))
* Fix missing `normalize_frequency` param in plots ([#213](https://github.com/MaartenGr/BERTopic/issues/208))
  

## **Version 0.9.0**
*Release date:  9 August, 2021*

**Highlights**:  

* Implemented a [**Guided BERTopic**](https://maartengr.github.io/BERTopic/getting_started/guided/guided.html) -> Use seeds to steer the Topic Modeling
* Get the most representative documents per topic: `topic_model.get_representative_docs(topic=1)`
    * This allows users to see which documents are good representations of a topic and better understand the topics that were created
* Added `normalize_frequency` parameter to `visualize_topics_per_class` and `visualize_topics_over_time` in order to better compare the relative topic frequencies between topics
* Return flat probabilities as default, only calculate the probabilities of all topics per document if `calculate_probabilities` is True
* Added several FAQs

**Fixes**:  

* Fix loading pre-trained BERTopic model
* Fix mapping of probabilities
* Fix [#190](https://github.com/MaartenGr/BERTopic/issues/190)


**Guided BERTopic**:    

Guided BERTopic works in two ways: 

First, we create embeddings for each seeded topics by joining them and passing them through the document embedder. 
These embeddings will be compared with the existing document embeddings through cosine similarity and assigned a label. 
If the document is most similar to a seeded topic, then it will get that topic's label. 
If it is most similar to the average document embedding, it will get the -1 label. 
These labels are then passed through UMAP to create a semi-supervised approach that should nudge the topic creation to the seeded topics. 

Second, we take all words in `seed_topic_list` and assign them a multiplier larger than 1. 
Those multipliers will be used to increase the IDF values of the words across all topics thereby increasing 
the likelihood that a seeded topic word will appear in a topic. This does, however, also increase the chance of an 
irrelevant topic having unrelated words. In practice, this should not be an issue since the IDF value is likely to 
remain low regardless of the multiplier. The multiplier is now a fixed value but may change to something more elegant, 
like taking the distribution of IDF values and its position into account when defining the multiplier. 

```python
seed_topic_list = [["company", "billion", "quarter", "shrs", "earnings"],
                   ["acquisition", "procurement", "merge"],
                   ["exchange", "currency", "trading", "rate", "euro"],
                   ["grain", "wheat", "corn"],
                   ["coffee", "cocoa"],
                   ["natural", "gas", "oil", "fuel", "products", "petrol"]]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
topics, probs = topic_model.fit_transform(docs)
```


## **Version 0.8.1**
*Release date:  8 June, 2021*

**Highlights**:  

* Improved models:
    * For English documents the default is now: `"paraphrase-MiniLM-L6-v2"` 
    * For Non-English or multi-lingual documents the default is now: `"paraphrase-multilingual-MiniLM-L12-v2"` 
    * Both models show not only great performance but are much faster!  
* Add interactive visualizations to the `plotting` API documentation
  
For better performance, please use the following models:  

* English: `"paraphrase-mpnet-base-v2"`
* Non-English or multi-lingual: `"paraphrase-multilingual-mpnet-base-v2"`

**Fixes**:   

* Improved unit testing for more stability
* Set transformers version for Flair

## **Version 0.8.0**
*Release date:  31 May, 2021*

**Highlights**:  

* Additional visualizations:
    * Topic Hierarchy: `topic_model.visualize_hierarchy()` 
    * Topic Similarity Heatmap: `topic_model.visualize_heatmap()` 
    * Topic Representation Barchart: `topic_model.visualize_barchart()` 
    * Term Score Decline: `topic_model.visualize_term_rank()` 
* Created `bertopic.plotting` library to easily extend visualizations
* Improved automatic topic reduction by using HDBSCAN to detect similar topics
* Sort topic ids by their frequency. -1 is the outlier class and contains typically the most documents. After that 0 is the largest  topic, 1 the second largest, etc. 
     
**Fixes**:   

* Fix typo [#113](https://github.com/MaartenGr/BERTopic/pull/113), [#117](https://github.com/MaartenGr/BERTopic/pull/117)
* Fix [#121](https://github.com/MaartenGr/BERTopic/issues/121) by removing [these](https://github.com/MaartenGr/BERTopic/blob/5c6cf22776fafaaff728370781a5d33727d3dc8f/bertopic/_bertopic.py#L359-L360) two lines
* Fix mapping of topics after reduction (it now excludes 0) ([#103](https://github.com/MaartenGr/BERTopic/issues/103))
  
## **Version 0.7.0**
*Release date:  26 April, 2021*  

The two main features are **(semi-)supervised topic modeling** 
and several **backends** to use instead of Flair and SentenceTransformers!

**Highlights**:

* (semi-)supervised topic modeling by leveraging supervised options in UMAP
    * `model.fit(docs, y=target_classes)`
* Backends:
    * Added Spacy, Gensim, USE (TFHub)
    * Use a different backend for document embeddings and word embeddings
    * Create your own backends with `bertopic.backend.BaseEmbedder`
    * Click [here](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html) for an overview of all new backends
* Calculate and visualize topics per class
    * Calculate: `topics_per_class = topic_model.topics_per_class(docs, topics, classes)`
    * Visualize: `topic_model.visualize_topics_per_class(topics_per_class)`
* Several tutorials were updated and added:

| Name  | Link  |
|---|---|
| Topic Modeling with BERTopic  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FieRA9fLdkQEGDIMYl0I3MCjSUKVF8C-?usp=sharing)  |
| (Custom) Embedding Models in BERTopic  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18arPPe50szvcCp_Y6xS56H2tY0m-RLqv?usp=sharing) |
| Advanced Customization in BERTopic  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ClTYut039t-LDtlcd-oQAdXWgcsSGTw9?usp=sharing) |
| (semi-)Supervised Topic Modeling with BERTopic  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bxizKzv5vfxJEB29sntU__ZC7PBSIPaQ?usp=sharing)  |
| Dynamic Topic Modeling with Trump's Tweets  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1un8ooI-7ZNlRoK0maVkYhmNRl0XGK88f?usp=sharing)  |

**Fixes**:  

* Fixed issues with Torch req
* Prevent saving term frequency matrix in CTFIDF class
* Fixed DTM not working when reducing topics ([#96](https://github.com/MaartenGr/BERTopic/issues/96))
* Moved visualization dependencies to base BERTopic
    * `pip install bertopic[visualization]` becomes `pip install bertopic`
* Allow precomputed embeddings in bertopic.find_topics() ([#79](https://github.com/MaartenGr/BERTopic/issues/79)):
 
```python
model = BERTopic(embedding_model=my_embedding_model)
model.fit(docs, my_precomputed_embeddings)
model.find_topics(search_term)
```

## **Version 0.6.0**
*Release date:  1 March, 2021*

**Highlights**:

* DTM: Added a basic dynamic topic modeling technique based on the global c-TF-IDF representation 
    * `model.topics_over_time(docs, timestamps, global_tuning=True)`
* DTM: Option to evolve topics based on t-1 c-TF-IDF representation which results in evolving topics over time
    * Only uses topics at t-1 and skips evolution if there is a gap
    * `model.topics_over_time(docs, timestamps, evolution_tuning=True)`
* DTM: Function to visualize topics over time 
    * `model.visualize_topics_over_time(topics_over_time)`
* DTM: Add binning of timestamps  
    * `model.topics_over_time(docs, timestamps, nr_bins=10)`
* Add function get general information about topics (id, frequency, name, etc.) 
    *  `get_topic_info()`
* Improved stability of c-TF-IDF by taking the average number of words across all topics instead of the number of documents

**Fixes**:

*  `_map_probabilities()` does not take into account that there is no probability of the outlier class and the probabilities are mutated instead of copied (#63, #64)

## **Version 0.5.0**
*Release date:  8 Februari, 2021*

**Highlights**:
  
* Add `Flair` to allow for more (custom) token/document embeddings, including 🤗 transformers 
* Option to use custom UMAP, HDBSCAN, and CountVectorizer
* Added `low_memory` parameter to reduce memory during computation
* Improved verbosity (shows progress bar)
* Return the figure of `visualize_topics()`
* Expose all parameters with a single function: `get_params()`

**Fixes**:
    
* To simplify the API, the parameters stop_words and n_neighbors were removed. These can still be used when a custom UMAP or CountVectorizer is used.
* Set `calculate_probabilities` to False as a default. Calculating probabilities with HDBSCAN significantly increases computation time and memory usage. Better to remove calculating probabilities or only allow it by manually turning this on.
* Use the newest version of `sentence-transformers` as it speeds ups encoding significantly

## **Version 0.4.2**
*Release date:  10 Januari, 2021*

**Fixes**:  

* Selecting `embedding_model` did not work when `language` was also used. This led to the user needing 
to set `language` to None before being able to use `embedding_model`. Fixed by using `embedding_model` when 
`language` is used (as a default parameter).

## **Version 0.4.1**
*Release date:  07 Januari, 2021*

**Fixes**:  

* Simple fix by lowering the languages variable to match the lowered input language.

## **Version 0.4.0**
*Release date:  21 December, 2020*

**Highlights**:  

* Visualize Topics similar to [LDAvis](https://github.com/cpsievert/LDAvis)
* Added option to reduce topics after training
* Added option to update topic representation after training
* Added option to search topics using a search term
* Significantly improved the stability of generating clusters
* Finetune the topic words by selecting the most coherent words with the highest c-TF-IDF values 
* More extensive tutorials in the documentation

**Notable Changes**:  

* Option to select language instead of sentence-transformers models to minimize the complexity of using BERTopic
* Improved logging (remove duplicates) 
* Check if BERTopic is fitted 
* Added TF-IDF as an embedder instead of transformer models (see tutorial)
* Numpy for Python 3.6 will be dropped and was therefore removed from the workflow.
* Preprocess text before passing it through c-TF-IDF
* Merged `get_topics_freq()` with `get_topic_freq()` 

**Fixes**:  

* Fix error handling topic probabilities

## **Version 0.3.2**
*Release date:  16 November, 2020*

**Highlights**:

* Fixed a bug with the topic reduction method that seems to reduce the number of topics but not to the nr_topics as defined in the class. Since this was, to a certain extend, breaking the topic reduction method a new release was necessary.

## **Version 0.3.1**
*Release date:  4 November, 2020*

**Highlights**:

* Adding the option to use custom embeddings or embeddings that you generated beforehand with whatever package you'd like to use. This allows users to further customize BERTopic to their liking.

## **Version 0.3.0**
*Release date:  29 October, 2020*

**Highlights**:

- transform() and fit_transform() now also return the topic probability distributions
- Added visualize_distribution() which visualizes the topic probability distribution for a single document

## **Version 0.2.2**
*Release date:  17 October, 2020*

**Highlights**:

- Fixed n_gram_range not being used
- Added option for using stopwords

## **Version 0.2.1**
*Release date:  11 October, 2020*

**Highlights**:

* Improved the calculation of the class-based TF-IDF procedure by limiting the calculation to sparse matrices. This prevents out-of-memory problems when faced with large datasets.

## **Version 0.2.0**  
*Release date:  11 October, 2020*

**Highlights**:

- Changed c-TF-IDF procedure such that it implements a version of scikit-learns procedure. This should also speed up the calculation of the sparse matrix and prevent memory errors.
- Added automated unit tests  

## **Version 0.1.2**
*Release date:  1 October, 2020*

**Highlights**:

* When transforming new documents, self.mapped_topics seemed to be missing. Added to the init.

## **Version 0.1.1**
*Release date:  24 September, 2020*

**Highlights**:

* Fixed requirements --> Issue with pytorch
* Update documentation

## **Version 0.1.0**  
*Release date:  24 September, 2020*

**Highlights**:  

- First release of `BERTopic`
- Added parameters for UMAP and HDBSCAN
- Option to choose sentence-transformer model
- Method for transforming unseen documents
- Save and load trained models (UMAP and HDBSCAN)
- Extract topics and their sizes

**Notable Changes**:  

- Optimized c-TF-IDF
- Improved documentation
- Improved topic reduction
 

