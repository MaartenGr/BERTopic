## Why are the results not consistent between runs?
Due to the stochastisch nature of UMAP, the results from BERTopic might differ even if you run the same code
multiple times. Using your own embeddings allows you to try out BERTopic several times until you find the 
topics that suit you best. You only need to generate the embeddings itself once and run BERTopic several times
with different parameters. 

## Which model is best suited to my language?
This is a difficult questions that I, unfortunately, cannot give you a perfect answer to, it depends...
However, the default model (i.e., distilbert-base-nli-stsb-mean-tokens) works great for English documents and by 
setting `language` to `multilingual` a model (i.e., xlm-r-bert-base-nli-stsb-mean-tokens) will be selected that 
works quite well for >50 languages. 

[SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) work typically quite well 
and are the preferred models to use. However, there are many 🤗 transformers available [here](https://huggingface.co/models) that could 
suit your use case. These models should be loaded in with Flair (see Guides/Embeddings). 

## How can I speed up BERTopic?
You can speed up BERTopic by either generating your embeddings beforehand, which is not advised, or by 
setting `calculate_probabilities` to False. Calculating the probabilities is quite expensive and can 
significantly increase the computation time. Thus, only use it if you do not mind waiting a bit before 
the model is done running or if you have less than 50_000 documents. 

## I am facing memory issues, help!
To prevent any memory issues, it is advised to set `low_memory` to True. This will result in UMAP being 
a bit slower, but consuming significantly less memory.

If the problem still persists, then this could be an issue related to your available memory. Processing 
millions of documents is quite computationally expensive and sufficient RAM is necessary.  

## I have only a few topics, how do I increase them?
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