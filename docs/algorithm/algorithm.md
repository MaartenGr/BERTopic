---
hide:
  - navigation
---

# The Algorithm

Below, you will find different types of overviews of each step in BERTopic's main algorithm. Each successive overview will be more in-depth than the previous overview. The aim of this approach is to make the underlying algorithm as intuitive as possible for a wide range of users. 

## **Visual Overview**

This visual overview reduces BERTopic to four main steps, namely the embedding of documents, the clustering of documents, the topic extraction, and the topic diversification. Each step contains one or more sub-steps that you can read a bit more about below. 
<br><br>
<script src="https://cdn.tailwindcss.com"></script>

<!-- <script src="https://unpkg.com/flowbite@1.5.2/dist/flowbite.js"></script> -->

<div class="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-x-4 w-4/5 
sm:w-full md:w-4/5 lg:w-4/5 xl:w-4/5 2xl:4/5">

  <!--Title stage 1-->
  <div class="overflow-hidden">
    <p class="text-blue-300 text-center text-lg">Embed documents</p>
  </div>
  <div class="overflow-hidden"></div>


  <!--Card 1-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
      <img class="h-48 pb-1" src="1. embeddings.svg" alt="embeddings">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
        <a href="/BERTopic/getting_started/embeddings/embeddings.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">1. Embeddings</a>
      </div>
      <p class="text-gray-700 text-sm">
        We start by converting our documents to vector representations through the use of language models. 
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">
        SBERT
      </span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">
        🤗 Transformers
      </span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">
        spaCy
      </span>
    </div>
  </div>
  <div class="overflow-hidden"></div>

  <!--Title stage 2-->
  <div class="overflow-hidden sm:col-span-1 md:col-span-2 lg:col-span-2 xl:col-span-2 pt-5">
  
  <p class="text-blue-300 text-center text-lg">Cluster embeddings</p>
  </div>

  <!--Card 2-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
        <img class="h-48 pb-1" src="2. reduction.svg" alt="reduction">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
      <a href="/BERTopic/getting_started/dim_reduction/dim_reduction.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">2. Dimension Reduction</a>
      </div>
      <p class="text-gray-700 text-sm">
        The vector representations are reduced in dimensionality so that clustering algorithms have an easier time finding clusters.
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">UMAP</span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">PCA</span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">Truncated SVD</span>
    </div>
  </div>

  <!--Card 3-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
        <img class="h-48 pb-1" src="3. cluster.svg" alt="cluster">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
      <a href="/BERTopic/getting_started/clustering/clustering.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">3. Clustering</a>
      </div>
      <p class="text-gray-700 text-sm">
        A clustering algorithm is used to cluster the reduced vectors in order to find semantically similar documents. 
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">HDBSCAN</span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">k-Means</span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">BIRCH</span>
    </div>
  </div>

  <div class="overflow-hidden sm:col-span-1 md:col-span-2 lg:col-span-2 xl:col-span-2 pt-5">
    <p class="text-blue-300 text-center text-lg">Topic Representation</p>
  </div>

  <!--Card 4-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
        <img class="h-48 pb-1" src="4. bow.svg" alt="bow">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
      <a href="/BERTopic/getting_started/countvectorizer/countvectorizer.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">4. Bag-of-Words</a>
      </div>
      <p class="text-gray-700 text-sm">
        We tokenize each topic into a bag-of-words representation that allows us to process the data without affecting the input embeddings.
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">CountVectorizer</span>
    </div>
  </div>

  <!--Card 5-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
        <img class="h-48 pb-1" src="5. ctfidf.svg" alt="ctfidf">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
      <a href="/BERTopic/getting_started/ctfidf/ctfidf.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">5. Topic Representation</a>
      </div>
      <p class="text-gray-700 text-sm">
        We calculate words that are interesting to each topic with a class-based TF-IDF procedure called c-TF-IDF.
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">c-TF-IDF</span>
      <span class="inline-block bg-slate-200 rounded-full px-2 py-1 text-xs text-gray-700 mr-1 mb-1">BM-25</span>
    </div>
  </div>

  <div class="overflow-hidden pt-5">
    <p class="text-blue-300 text-center text-lg">(Optional) Topic Diversity</p>
  </div>
  <div class="overflow-hidden"></div>

  <!--Card 6-->
  <div class="rounded overflow-hidden shadow-outline shadow-md shadow-slate-300
              border-solid border-2 border-slate-100">
    <div class="flex justify-center pt-4">
        <img class="h-48 pb-1" src="6. diversity.svg" alt="embeddings">
    </div>
    <div class="px-6">
      <div class="font-bold text-lg mb-2">
      <a href="/BERTopic/getting_started/diversity/diversity.html" class="text-sky-800 no-underline hover:underline hover:text-sky-800">6. Topic Diversity</a>
      </div>
      <p class="text-gray-700 text-sm">
        Maximal Marginal Relevance is used to diversify words in each topic which removes repeating and similar words. 
      </p>
    </div>
    <div class="px-6 pt-4 pb-2">
      <span class="inline-block bg-slate-200 rounded-full px-3 py-2 text-xs text-gray-700 mr-1 mb-1">MMR</span>
    </div>
  </div>

</div>

<script>
</script>

## **Code Overview**
After going through the visual overview, this code overview demonstrates the algorithm using BERTopic. An advantage of using BERTopic is each major step in its algorithm can be explicitly defined, thereby making the process not only transparent but also more intuitive. 


```python
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,    # Step 1 - Extract embeddings
  umap_model=umap_model,              # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
  diversity=0.5                       # Step 6 - Diversify topic words
)
```


## **Detailed Overview**
This overview describes each step in more detail such that you can get an intuitive feeling as to what models might fit best at each step in your use case. 

###  **1. Embed documents**
We start by converting our documents to numerical representations. Although there are many methods for doing so the default in BERTopic is [sentence-transformers](https://github.com/UKPLab/sentence-transformers). These models are often optimized for semantic similarity which helps tremendously in our clustering task. Moreover, they are great for creating either document- or sentence-embeddings. 
<br>
In BERTopic, you can choose any sentence-transformers model but there are two models that are set as defaults:

* `"all-MiniLM-L6-v2"`
* `"paraphrase-multilingual-MiniLM-L12-v2"`

The first is an English language model trained specifically for semantic similarity tasks which work quite 
well for most use-cases. The second model is very similar to the first with one major difference is that the 
`multilingual` models work for 50+ languages. This model is quite a bit larger than the first and is only selected if 
you select any language other than English.

!!! tip Embedding models

    Although BERTopic uses sentence-transformers models as a default, you can choose 
    any embedding model that fits your use case. Follow the guide [here](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html) for selecting 
    and customizing your model.


### **2. Dimensionality reduction**
After having created our numerical representations of the documents we have to reduce the dimensionality of these representations. Cluster models typically have difficulty handling high dimensional data due to the curse of dimensionality. There are great approaches that can reduce dimensionality, such as PCA, but as a default [UMAP](https://github.com/lmcinnes/umap) is selected in BERTopic. It is a technique that can keep some of a dataset's local and global structure when reducing its dimensionality. This structure is important to keep as it contains the information necessary to create clusters of semantically similar documents. 

!!! tip Dimensionality reduction models

    Although BERTopic uses UMAP as a default, you can choose 
    any dimensionality reduction model that fits your use case. Follow the guide [here](https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html) for selecting 
    and customizing your model.

###  **3. Cluster Documents**
After having reduced our embeddings, we can start clustering our data. For that, we leverage a density-based clustering technique, HDBSCAN. It can find clusters of different shapes and has the nice feature of identifying outliers where possible. As a result, we do not force documents in a cluster where they might note belong. This will improve the resulting topic representation as there is less noise to draw from. 

!!! tip Cluster models

    Although BERTopic uses HDBSCAN as a default, you can choose 
    any cluster model that fits your use case. Follow the guide [here](https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html) for selecting 
    and customizing your model.

### **4. Bag-of-words**
Before we can start creating the topic representation we first need to select a technique that allows for modularity in BERTopic's algorithm. When we use HDBSCAN as a cluster model, we may assume that our clusters having different degrees of density and different shapes. This means that a centroid-based topic representation technique might not be the best fitting model. In other words, we want a topic representation technique that makes little to no assumption on the expected structure of the clusters. 
<br>
To do this, we first combine all documents in a cluster into a single document. That, very long, document then represents the cluster. Then, we can count how often each word appears in each cluster. This generates something called a bag-of-words representation in which resides the frequency of each word in each cluster. This bag-of-words representation is therefore on a cluster-level and not on a document-level. This distinction is important as we are interested in words on a topic-level (i.e., cluster-level). By using a bag-of-words representation, no assumption is made with respect to the structure of the clusters. Moreover, the bag-of-words representation is L1-normalized to account for clusters that have different sizes. 

!!! tip Bag-of-words and tokenization

    There are many ways you can tune or change the bag-of-words step. This step allows for processing the documents however you want without affecting the first step, embedding the documents. You can follow the guide [here](https://maartengr.github.io/BERTopic/getting_started/countvectorizer/countvectorizer.html) for more information about tokenization options in BERTopic.

###  **5. Topic representation**
From the generated bag-of-words representation, we want to know what makes one cluster different from another? Which words are typical for cluster 1 and not so much for all other clusters? To solve this, we need to modify TF-IDF such that it considers topics (i.e., clusters) instead of documents. 
<br>    
When you apply TF-IDF as usual on a set of documents, what you are doing is comparing the importance of 
words between documents. Now, what if, we instead treat all documents in a single category (e.g., a cluster) as a single document and then apply TF-IDF? The result would be importance scores for words within a cluster. The more important words are within a cluster, the more it is representative of that topic. In other words, if we extract the most important words per cluster, we get descriptions of **topics**! This model is called **class-based TF-IDF**:
<br><br>
  
<img class="w-6/12" src="c-TF-IDF.svg">

<br>
Each cluster is converted to a single document instead of a set of documents. Then, we extract the frequency of word `x` in class `c`, where `c` refers to the cluster we created before. This results in our class-based `tf` representation. This representation is L1-normalized to account for the differences in topic sizes. 
  <br><br>
Then, we take take the logarithm of one plus the average number of words per class `A` divided by the frequency of word `x` across all classes. We add plus one within the logarithm to force values to be positive. This results in our class-based `idf` representation. Like with the classic TF-IDF, we then multiply `tf` with `idf` to get the importance score per word in each class. In other words, the classical TF-IDF procedure is **not** used here but a modified version of the algorithm that allows for a much better representation. 

!!! tip c-TF-IDF parameters

    In the `ClassTfidfTransformer`, there are a few parameters that might be worth exploring, including an option to perform additional BM-25 weighting. You can find more information about that [here](https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html).

### **6. (Optional) Maximal Marginal Relevance**  
After having generated the c-TF-IDF representations, we have a set of words that describe a collection of documents. 
Technically, this does not mean that this collection of words describes a coherent topic. In practice, we will 
see that many of the words do describe a similar topic but some words will, in a way, overfit the documents. For 
example, if you have a set of documents that are written by the same person whose signature will be in the topic 
description. 
<br>
To improve the coherence of words, Maximal Marginal Relevance was used to find the most coherent words 
without having too much overlap between the words themselves. This results in the removal of words that do not contribute 
to a topic.  
<br>
You can also use this technique to diversify the words generated in the topic representation. At times, many variations 
of the same word can end up in the topic representation. To reduce the number of synonyms, we can increase the diversity 
among words whilst still being similar to the topic representation. 
