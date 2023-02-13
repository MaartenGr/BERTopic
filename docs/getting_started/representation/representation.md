One of the core components of BERTopic is its Bag-of-Words representation and weighting with c-TF-IDF. This method is fast and can quickly generate a number of keywords for a topic without depending on the clustering task. As a result, topics can easily and quickly be updated after training the model without the need to re-train it. 
Although these give good topic representations, we may want to further fine-tune the topic representations. 

As such, there are a number of representation models implemented in BERTopic that allows for further fine-tuning of the topic representations. These are optional 
and are **not used by default**. You are not restrained by the how the representation can be fine-tuned, from GPT-like models to fast keyword extraction 
with KeyBERT-like models:

<iframe width="1200" height="500" src="https://user-images.githubusercontent.com/25746895/218417067-a81cc179-9055-49ba-a2b0-f2c1db535159.mp4
" title="BERTopic Overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For each model below, an example will be shown on how it may change or improve upon the default topic keywords that are generated. The dataset used in these examples can be found [here](https://www.kaggle.com/datasets/maartengr/kurzgesagt-transcriptions). 

## **KeyBERTInspired**

After having generated our topics with c-TF-IDF, we might want to do some fine-tuning based on the semantic
relationship between keywords/keyphrases and the set of documents in each topic. Although we can use a centroid-based
technique for this, it can be costly and does not take the structure of a cluster into account. Instead, we leverage 
c-TF-IDF to create a set of representative documents per topic and use those as our updated topic embedding. Then, we calculate 
the similarity between candidate keywords and the topic embedding using the same embedding model that embedded the documents. 

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/keybertinspired.svg"
</div>
<br>

Thus, the algorithm follows some principles of [KeyBERT](https://github.com/MaartenGr/KeyBERT) but does some optimization in
order to speed up inference. Usage is straightforward:

```python
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic

# Create your representation model
representation_model = KeyBERTInspired()

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/keybert.svg"
</div>
<br>

## **PartOfSpeech**
Our candidate topics, as extracted with c-TF-IDF, do not take into account a keyword's part of speech as extracting noun-phrases from 
all documents can be computationally quite expensive. Instead, we can leverage c-TF-IDF to perform part of speech on a subset of 
keywords and documents that best represent a topic. 

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/partofspeech.svg"
</div>
<br>

More specifically, we find documents that contain the keywords from our candidate topics as calculated with c-TF-IDF. These documents serve 
as the representative set of documents from which the Spacy model can extract a set of candidate keywords for each topic.
These candidate keywords are first put through Spacy's POS module to see whether they match with the `DEFAULT_PATTERNS`:

```python
DEFAULT_PATTERNS = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}],
            [{'POS': 'ADJ'}]
]
```

These patterns follow Spacy's [Rule-Based Matching](https://spacy.io/usage/rule-based-matching). Then, the resulting keywords are sorted by 
their respective c-TF-IDF values.

```python
from bertopic.representation import PartOfSpeech
from bertopic import BERTopic

# Create your representation model
representation_model = PartOfSpeech("en_core_web_sm")

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/pos.svg"
</div>
<br>

You can define custom POS patterns to be extracted:

```python
pos_patterns = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}], [{'POS': 'ADJ'}]
]
representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns)
```


## **MaximalMarginalRelevance**
When we calculate the weights of keywords, we typically do not consider whether we already have similar keywords in our topic. Words like "car" and "cars" 
essentially represent the same information and often redundant. 

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/mmr.svg"
</div>
<br>

<!-- MMR = arg  \underset{D_i\in R\setminus S}{max} [\lambda Sim_{1}(D_{i}, Q) - (1-\lambda) \,\, \underset{D_{j}\in S}{max} \,\, Sim_{2}(D_{i}, D_{j})] -->

To decrease this redundancy and improve the diversity of keywords, we can use an algorithm called Maximal Marginal Relevance (MMR). MMR considers the similarity of keywords/keyphrases with the document, along with the similarity of already selected keywords and keyphrases. This results in a selection of keywords
that maximize their within diversity with respect to the document.


```python
from bertopic.representation import MaximalMarginalRelevance
from bertopic import BERTopic

# Create your representation model
representation_model = MaximalMarginalRelevance(diversity=0.3)

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/mmr_output.svg"
</div>
<br>

## **Zero-Shot Classification**

For some use cases, you might already have a set of candidate labels that you would like to automatically assign to some of the topics. 
Although we can use guided or supervised BERTopic for that, we can also use zero-shot classification to assign labels to our topics. 
For that, we can make use of 🤗 transformers on their models on the [model hub](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads). 

To perform this classification, we feed the model with the keywords as generated through c-TF-IDF and a set of candidate labels. 
If, for a certain topic, we find a similar enough label, then it is assigned. If not, then we keep the original c-TF-IDF keywords. 

We use it in BERTopic as follows:

```python
from bertopic.representation import ZeroShotClassification
from bertopic import BERTopic

# Create your representation model
candidate_topics = ["space and nasa", "bicycles", "sports"]
representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/zero.svg"
</div>
<br>

## **Text Generation**

Text generation models, like GPT-3 and the well-known ChatGPT, are becoming more and more capable of generating sensible output. 
For that purpose, a number of models are exposed in BERTopic that allow topic labels to be created based on candidate documents and keywords
for each topic. These candidate documents and keywords are created from BERTopic's c-TF-IDF calculation. A huge benefit of this is that we can
describe a topic with only a few documents and we therefore do not need to pass all documents to the text generation model. Not only speeds 
this the generation of topic labels up significantly, you also do not need a massive amount of credits when using an external API, such as Cohere or OpenAI.


!!! tip Tip
    You can access the default prompts of these models with `representation_model.default_prompt_`

### **🤗 Transformers**

Nearly every week, there are new and improved models released on the 🤗 [Model Hub](https://huggingface.co/models) that, with some creativity, allow for 
further fine-tuning of our c-TF-IDF based topics. These models range from text generation to zero-classification. In BERTopic, wrappers around these 
methods are created as a way to support whatever might be released in the future. 

Using a GPT-like model from the huggingface hub is rather straightforward:

```python
from bertopic.representation import TextGeneration
from bertopic import BERTopic

# Create your representation model
representation_model = TextGeneration('gpt2')

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

You can use a custom prompt and decide where the keywords should
be inserted by using the `[KEYWORDS]` or documents with the `[DOCUMENTS]` tag:

```python
from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?""

# Create your representation model
generator = pipeline('text2text-generation', model='google/flan-t5-base')
representation_model = TextGeneration(generator)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/hf.svg"
</div>
<br>

As can be seen from the example above, if you would like to use a `text2text-generation` model, you will to 
pass a `transformers.pipeline` with the `"text2text-generation"` parameter. 


### **Cohere**

Instead of using a language model from 🤗 transformers, we can use external APIs instead that 
do the work for you. Here, we can use [Cohere](https://docs.cohere.ai/) to extract our topic labels from the candidate documents and keywords.
To use this, you will need to install cohere first:

```bash
pip install cohere
```

Then, get yourself an API key and use Cohere's API as follows:

```python
import cohere
from bertopic.representation import Cohere
from bertopic import BERTopic

# Create your representation model
co = cohere.Client(my_api_key)
representation_model = Cohere(co)

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/cohere.svg"
</div>
<br>

You can also use a custom prompt:

```python
prompt = """
I have topic that contains the following documents: [DOCUMENTS]. 
The topic is described by the following keywords: [KEYWORDS].
Based on the above information, can you give a short label of the topic?
"""
representation_model = Cohere(co, prompt=prompt)
```

### **OpenAI**

Instead of using a language model from 🤗 transformers, we can use external APIs instead that 
do the work for you. Here, we can use [OpenAI](https://openai.com/api/) to extract our topic labels from the candidate documents and keywords.
To use this, you will need to install openai first:

`pip install openai`

Then, get yourself an API key and use OpenAI's API as follows:

```python
import openai
from bertopic.representation import OpenAI
from bertopic import BERTopic

# Create your representation model
openai.api_key = MY_API_KEY
representation_model = OpenAI()

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/openai.svg"
</div>
<br>

You can also use a custom prompt:

```python
prompt = "I have the following documents: [DOCUMENTS]. What topic do they contain?"
representation_model = OpenAI(prompt=prompt)
```

### **LangChain**

[Langchain](https://github.com/hwchase17/langchain) is a package that helps users with chaining large language models.
In BERTopic, we can leverage this package in order to more efficiently combine external knowledge. Here, this 
external knowledge are the most representative documents in each topic. 

To use langchain, you will need to install the langchain package first. Additionally, you will need an underlying LLM to support langchain,
like openai:

```bash
pip install langchain, openai
```

Then, you can create your chain as follows:

```python
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=my_openai_api_key), chain_type="stuff")
```

Finally, you can pass the chain to BERTopic as follows:

```python
from bertopic.representation import LangChain

# Create your representation model
representation_model = LangChain(chain)

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

You can also use a custom prompt:

```python
prompt = "What are these documents about? Please give a single label."
representation_model = LangChain(chain, prompt=prompt)
```

!!! note Note
    The prompt does not make use of `[KEYWORDS]` and `[DOCUMENTS]` tags as 
    the documents are already used within langchain's `load_qa_chain`. 


## **Chain Models**

All of the above models can make use of the candidate topics, as generated by c-TF-IDF, to further fine-tune the topic representations. For example, `MaximalMarginalRelevance` takes the keywords in the candidate topics and re-ranks them. Similarly, the keywords in the candidate topic can be used as the input for GPT-prompts in `OpenAI`.

Although the default candidate topics are generated by c-TF-IDF, what if we were to chain these models? For example, we can use `MaximalMarginalRelevance` to improve upon the keywords in each topic before passing them to `OpenAI`. 

This is supported in BERTopic by simply passing a list of representation models when instantation the topic model:

```python
from bertopic.representation import MaximalMarginalRelevance, OpenAI
from bertopic import BERTopic
import openai

# Create your representation models
openai.api_key = MY_API_KEY
openai_generator = OpenAI()
mmr = MaximalMarginalRelevance(diversity=0.3)
representation_models = [mmr, openai_generator]

# Use the chained models
topic_model = BERTopic(representation_model=representation_models)
```

## **Custom Model**

Although several representation models have been implemented in BERTopic, new technologies get released often and we should not have to wait until they get implemented in BERTopic. Therefore, you can create your own representation model and use that to fine-tune the topics. 

The following is the basic structure for creating your custom model. Note that it returns the same topics as the those 
calculated with c-TF-IDF:

```python
from bertopic.representation._base import BaseRepresentation


class CustomRepresentationModel(BaseRepresentation):
    def __init__(self):
        pass

    def extract_topics(self, topic_model, documents, c_tf_idf, topics
                      ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: The BERTopic model
            documents: A dataframe of documents with their related topics
            c_tf_idf: The c-TF-IDF matrix
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        updated_topics = topics.copy()
        return updated_topics
```

Then, we can use that model as follows:

```python
from bertopic import BERTopic

# Create our custom representation model
representation_model = CustomRepresentationModel()

# Pass our custom representation model to BERTopic
topic_model = BERTopic(representation_model=representation_model)
```

There are a few things to take into account when creating your custom model:

* It needs to have the exact same parameter input: `topic_model`, `documents`, `c_tf_idf`, `topics`.
* You can change the `__init__` however you want, it does not influence the underlying structure
* Make sure that `updated_topics` has the exact same structure as `topics`:
    * For example: `updated_topics = {"1", [("space", 0.9), ("nasa", 0.7)], "2": [("science", 0.66), ("article", 0.6)]`}
    * Thus, it is a dictionary where each topic is represented by a list of keyword,value tuples.
* Lastly, make sure that `updated_topics` contains at least 5 keywords, even if they are empty: `[("", 0), ("", 0), ...]`