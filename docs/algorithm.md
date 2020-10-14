# The Algorithm  
The algorithm contains, roughly, 3 stages:   
   
- Extract document embeddings with **Sentence Transformers**  
- Cluster document embeddings to create groups of similar documents with **UMAP** and **HDBSCAN**  
- Extract and reduce topics with **c-TF-IDF**  

##  Sentence Transformer
We start by creating document embeddings from a set of documents using 
[sentence-transformer](https://github.com/UKPLab/sentence-transformers). These models are pre-trained for many 
language and are great for creating either document- or sentence-embeddings. 

If you have long documents, I would advise you to split up your documents into paragraphs or sentences as a BERT-based
model in `sentence-transformer` typically has a token limit. 

##  UMAP + HDBSCAN
Next, in order to cluster the documents using a clustering algorithm such as HDBSCAN we first need to 
reduce its dimensionality as HDBCAN is prone to the curse of dimensionality.

<p align="center">
<img src="https://github.com/MaartenGr/BERTopic/raw/master/images/clusters.png"/>
</p>

Thus, we first lower dimensionality with UMAP as it preserves local structure well after which we can 
use HDBSCAN to cluster similar documents.  

##  c-TF-IDF
What we want to know from the clusters that we generated, is what makes one cluster, based on their content, 
different from another? To solve this, we can modify TF-IDF such that it allows for interesting words per topic
instead of per document. 

When you apply TF-IDF as usual on a set of documents, what you are basically doing is comparing the importance of 
words between documents. Now, what if, we instead treat all documents in a single category (e.g., a cluster) 
as a single document and then apply TF-IDF? The result would be importance scores for words within a cluster. 
The more important words are within a cluster, the more it is representative of that topic. In other words, 
if we extract the most important words per cluster, we get descriptions of **topics**! 

<p align="center">
<img src="https://github.com/MaartenGr/BERTopic/raw/master/images/ctfidf.png" height="50"/>
</p>  

Each cluster is converted to a single document instead of a set of documents. 
Then, the frequency of word `t` are extracted for each class `i` and divided by the total number of words `w`. 
This action can now be seen as a form of regularization of frequent words in the class.
Next, the total, unjoined, number of documents `m` is divided by the total frequency of word `t` across all classes `n`.