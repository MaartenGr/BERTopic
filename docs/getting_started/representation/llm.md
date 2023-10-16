As we have seen in the [previous section](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html), the topics that you get from BERTopic can be fine-tuned using a number of approaches. Here, we are going to focus on text generation Large Language Models such as ChatGPT, GPT-4, and open-source solutions. 

Using these techniques, we can further fine-tune topics to generate labels, summaries, poems of topics, and more. To do so, we first generate a set of keywords and documents that describe a topic best using BERTopic's c-TF-IDF calculate. Then, these candidate keywords and documents are passed to the text generation model and asked to generate output that fits the topic best. 

A huge benefit of this is that we can describe a topic with only a few documents and we therefore do not need to pass all documents to the text generation model. Not only speeds this the generation of topic labels up significantly, you also do not need a massive amount of credits when using an external API, such as Cohere or OpenAI.


## **Prompts**

In most of the examples below, we use certain tags to customize our prompts. There are currently two tags, namely `"[KEYWORDS]"` and `"[DOCUMENTS]"`. 
These tags indicate where in the prompt they are to be replaced with a topics keywords and top 4 most representative documents respectively. 
For example, if we have the following prompt:

```python
prompt = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""
```

then that will be rendered as follows:

```python
"""
I have a topic that contains the following documents: 
- Our videos are also made possible by your support on patreon.co.
- If you want to help us make more videos, you can do so on patreon.com or get one of our posters from our shop.
- If you want to help us make more videos, you can do so there.
- And if you want to support us in our endeavor to survive in the world of online video, and make more videos, you can do so on patreon.com.

The topic is described by the following keywords: videos video you our support want this us channel patreon make on we if facebook to patreoncom can for and more watch 

Based on the above information, can you give a short label of the topic?
"""
```

!!! tip "Tip 1"
    You can access the default prompts of these models with `representation_model.default_prompt_`. The prompts that were generated after training can be accessed with `topic_model.representation_model.prompts_`.

### **Selecting Documents**

By default, four of the most representative documents will be passed to `[DOCUMENTS]`. These documents are selected by calculating their similarity (through c-TF-IDF representations) with the main c-TF-IDF representation of the topics. The four best matching documents per topic are selected. 

To increase the number of documents passed to `[DOCUMENTS]`, we can use the `nr_docs` parameter which is accessible in all LLMs on this page. Using this value allows you to select the top *n* most representative documents instead. If you have a long enough context length, then you could even give the LLM dozens of documents.

However, some of these documents might be very similar to one another and might be near duplicates. They will not provide much additional information about the content of the topic. Instead, we can use the `diversity` parameter in each LLM to only select documents that are sufficiently diverse. It takes values between 0 and 1 but a value of 0.1 already does wonders!

### **Truncating Documents**

If you increase the number of documents passed to `[DOCUMENTS]`, then the token limit can quickly be filled. Instead, we could only pass a part of the document by truncating it to a specific length. For that, we use two parameters that are accessible in all LLMs on this page, namely `document_length` and `tokenizer`. 

Let's start with the `tokenizer`. It is used to split a document up into tokens/segments. Each segment can then be used to calculate the complete document length. 
The methods for tokenization are as follows:

* If tokenizer is `char`, then the document is split up into characters.
* If tokenizer is `whitespace`, the the document is split up into words separated by whitespaces. 
* If tokenizer is `vectorizer`, then the internal CountVectorizer is used to tokenize the document.
* If tokenizer is a `callable`, then that callable is used to tokenize the document.

After having tokenized the document according one of the strategies above, the `document_length` is used to truncate the document to its specified value. 

For example, if the tokenizer is `whitespace` then a document is split up into individual words and the length of the document is counted by the total number of words. In contrast, if the tokenizer is `callable` we can use any callable that has an `.encode` and `.decode` function. If we were to use [tiktoken](https://github.com/openai/tiktoken), then the document would be split up into tokens and the length of the document is counted by the total number tokens.

To give an example, using tiktoken would work as follows:

```python
import openai
import tiktoken
from bertopic.representation import OpenAI
from bertopic import BERTopic

# Create tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Create your representation model
openai.api_key = MY_API_KEY
representation_model = OpenAI(tokenizer=tokenizer, document_length=50)
```

In this example, each document will be at most 50 tokens, anything more will get truncated.

## **🤗 Transformers**

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

GPT2, however, is not the most accurate model out there on HuggingFace models. You can get 
much better results with a `flan-T5` like model:

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
pass a `transformers.pipeline` with the `"text2text-generation"` parameter. Moreover, you can use a custom prompt and decide where the keywords should
be inserted by using the `[KEYWORDS]` or documents with the `[DOCUMENTS]` tag.

### **Zephyr** (Mistral 7B)

We can go a step further with open-source Large Language Models (LLMs) that have shown to match the performance of closed-source LLMs like ChatGPT.

In this example, we will show you how to use Zephyr, a fine-tuning version of Mistral 7B. Mistral 7B outperforms other open-source LLMs at a much smaller scale and is a worthwhile solution for use cases such as topic modeling. We want to keep inference as fast as possible and a relatively small model helps with that. Zephyr is a fine-tuned version of Mistral 7B that was trained on a mix of publicly available and synthetic datasets using Direct Preference Optimization (DPO).

To use Zephyr in BERTopic, we will first need to install and update a couple of packages that can handle quantized versions of Zephyr:

```python
pip install ctransformers[cuda]
pip install --upgrade git+https://github.com/huggingface/transformers
```

Instead of loading in the full model, we can instead load a quantized model which is a compressed version of the original model:

```python
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-alpha-GGUF",
    model_file="zephyr-7b-alpha.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50,
    hf=True
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

# Pipeline
generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)
```

This Zephyr model requires a specific prompt template in order to work:

```python
prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
<|user|>
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""
```

After creating this prompt template, we can create our representation model to be used in BERTopic:


```python
from bertopic.representation import TextGeneration

# Text generation with Zephyr
zephyr = TextGeneration(generator, prompt=prompt)
representation_model = {"Zephyr": zephyr}

# Topic Modeling
topic_model = BERTopic(representation_model=representation_model, verbose=True)
```

From 

## **OpenAI**

Instead of using a language model from 🤗 transformers, we can use external APIs instead that 
do the work for you. Here, we can use [OpenAI](https://openai.com/api/) to extract our topic labels from the candidate documents and keywords.
To use this, you will need to install openai first:

```bash
pip install openai
```

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
prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
representation_model = OpenAI(prompt=prompt)
```

### **ChatGPT**

Within OpenAI's API, the ChatGPT models use a different API structure compared to the GPT-3 models. 
In order to use ChatGPT with BERTopic, we need to define the model and make sure to enable `chat`:

```python
representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)
```

Prompting with ChatGPT is very satisfying and is customizable as follows:

```python
prompt = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""
```

!!! note 
    Whenever you create a custom prompt, it is important to add 
    ```
    Based on the information above, extract a short topic label in the following format:
    topic: <topic label>
    ```
    at the end of your prompt as BERTopic extracts everything that comes after `topic: `. Having 
    said that, if `topic: ` is not in the output, then it will simply extract the entire response, so 
    feel free to experiment with the prompts. 

### **Summarization**

Due to the structure of the prompts in OpenAI's chat models, we can extract different types of topic representations from their GPT models. 
Instead of extracting a topic label, we can instead ask it to extract a short description of the topic instead:

```python
summarization_prompt = """
I have a topic that is described by the following keywords: [KEYWORDS]
In this topic, the following documents are a small but representative subset of all documents in the topic:
[DOCUMENTS]

Based on the information above, please give a description of this topic in the following format:
topic: <description>
"""

representation_model = OpenAI(model="gpt-3.5-turbo", chat=True, prompt=summarization_prompt, nr_docs=5, delay_in_seconds=3)
```

The above is not constrained to just creating a short description or summary of the topic, we can extract labels, keywords, poems, example documents, extensitive descriptions, and more using this method!
If you want to have multiple representations of a single topic, it might be worthwhile to also check out [**multi-aspect**](https://maartengr.github.io/BERTopic/getting_started/multiaspect/multiaspect.html) topic modeling with BERTopic.


## **LangChain**

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

## **Cohere**

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
I have topic that contains the following documents: [DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS].
Based on the above information, can you give a short label of the topic?
"""
representation_model = Cohere(co, prompt=prompt)
```