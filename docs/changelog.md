## **Version 0.5.0**
*Release date:  XXX*

**Highlights**:

* Added `low_memory` parameter to reduce memory during computation (may speed-up computation, does not calculate probabilities)
* Get parameters with `get_params()`
* Option to use custom UMAP
* Option to use custom HDBSAN
* Improved verbosity (shows progress bar)
* Use the newest version of sentence-transformers as it speeds ups encoding significantly
* Improved testing   

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
 

