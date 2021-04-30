## **Why are the results not consistent between runs?**
Due to the stochastic nature of UMAP, the results from BERTopic might differ even if you run the same code
multiple times. Using your own embeddings allows you to try out BERTopic several times until you find the 
topics that suit you best. You only need to generate the embeddings itself once and run BERTopic several times
with different parameters. 

## **Which embedding model works best for which language?**
Unfortunately, there is not a definitive list on the best models for each language, this highly depends 
on your data, the model, and your specific use-case. However, the default model in BERTopic 
(`"distilbert-base-nli-stsb-mean-tokens"`) works great for **English** documents. In contrast, for **multi-lingual** 
documents or any other language, `"xlm-r-bert-base-nli-stsb-mean-tokens""` has shown great performance.  

**SentenceTransformers**  
[SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) work typically quite well 
and are the preferred models to use. They are great in generating document embeddings and have several 
multi-lingual versions available.  

**🤗 transformers**  
BERTopic allows you to use any 🤗 transformers model. These models  are typically embeddings created on 
a word/sentence level but can easily be pooled using Flair (see Guides/Embeddings). If you have a 
specific language for which you want to generate embeddings, you can choose the model [here](https://huggingface.co/models).

## **How can I speed up BERTopic?**
You can speed up BERTopic by either generating your embeddings beforehand, which is not advised, or by 
setting `calculate_probabilities` to False. Calculating the probabilities is quite expensive and can 
significantly increase the computation time. Thus, only use it if you do not mind waiting a bit before 
the model is done running or if you have less than 50_000 documents. 

## **I am facing memory issues. Help!**
To prevent any memory issues, it is advised to set `low_memory` to True. This will result in UMAP being 
a bit slower, but consuming significantly less memory. Moreover, calculating the probabilities of topics 
is quite computationally consuming and might impact memory. Setting `calculate_probabilities` to False 
could similarly help. 

If the problem still persists, then this could be an issue related to your available memory. Processing 
millions of documents is quite computationally expensive and sufficient RAM is necessary.  

## **I have only a few topics, how do I increase them?**
There are several reasons why your topic model results in only a few topics. 

First, you might only have a few documents (~1000). This makes it very difficult to properly 
extract topics due to the little amount of data available. Increasing the number of documents 
might solve your issues. 

Second, `min_topic_size` might be simply too large for your number of documents. If you decrease 
the minimum size of topics, then you are much more likely to increase the number of topics generated.
You could also decrease the `n_neighbors` parameter used in `UMAP` if this does not work. 

Third, although this does not happen very often, there simply aren't that many topics to be found 
in your documents. You can often see this when you have many `-1` topics, which is actually not a topic 
but a category of outliers.  

## **Why are the probabilities not calculated?**
Although it is possible to calculate the probabilities, the process of doing so is quite computationally 
inefficient and might significantly increase the computation time. To prevent this, the probabilities are 
not calculated as a default. In order to calculate, you will have to set `calculate_probabilities` to True:

```python
from bertopic import BERTopic
topic_model = BERTopic(calculate_probabilities=True)
topics, probs = topic_model.fit_transform(docs) 
```  

## **Numpy gives me an error when running BERTopic**
With the release of Numpy 1.20.0, there have been significant issues with using that version (and previous) due 
to compilation issues and pypi.   
  
This is a known issue with the order of install using pypi. You can find more details about this issue 
[here](https://github.com/lmcinnes/umap/issues/567) and [here](https://github.com/scikit-learn-contrib/hdbscan/issues/457).

I would suggest doing one of the following:

* Install the newest version from BERTopic (>= v0.5).
* You can install hdbscan with pip install hdbscan --no-cache-dir --no-binary :all: --no-build-isolation which might resolve the issue
* Use the above step also with numpy as it is part of the issue
* Install BERTopic in a fresh environment using these steps. 


## **Can I use the GPU to speed up the model?**
Yes and no. The GPU is automatically used when you use a SentenceTransformer or Flair embedding model. Using a CPU 
would then definitely slow things down. However, UMAP and HDBSCAN are not GPU-accelerated and are likely not so in 
the near future. For now, a GPU does help tremendously for extracting embeddings but does not speed up all 
aspects of BERtopic.   

## **Should I preprocess the data?**
No. By using document embeddings there is typically no need to preprocess the data as all parts of a document 
are important in understanding the general topic of the document. Although this holds true in 99% of cases, if you 
have data that contains a lot of noise, for example HTML-tags, then it would be best to remove them. HTML-tags 
typically do not contribute to the meaning of a document and should therefore be removed. However, if you apply 
topic modeling to HTML-code to extract topics of code, then it becomes important.

## **Why does it take so long to import BERTopic?**
The main culprit here seems to be UMAP. After running thorough tests with [Tuna](https://github.com/nschloe/tuna) we 
can see that most of the resources when importing BERTopic can be dedicated to UMAP:   

<img src="img/tuna.png" />

Unfortunately, there currently is no fix for this issue. The most recent ticket regarding this 
issue can be found [here](https://github.com/lmcinnes/umap/issues/631).
 
