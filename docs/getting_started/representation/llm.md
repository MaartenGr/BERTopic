As we have seen in the [previous section](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html), the topics that you get from BERTopic can be fine-tuned using a number of approaches. Here, we are going to focus on text generation Large Language Models such as OpenAI's GPT models and open-source models such as Gemma.

Using these techniques, we can further fine-tune topics to generate labels, summaries, poems of topics, and more. To do so, we first generate a set of keywords and documents that describe a topic best using BERTopic's c-TF-IDF calculate. Then, these candidate keywords and documents are passed to the text generation model and asked to generate output that fits the topic best.

A huge benefit of this is that we can describe a topic with only a few documents and we therefore do not need to pass all documents to the text generation model. Not only speeds this the generation of topic labels up significantly, you also do not need a massive amount of credits when using an external API, such as Cohere or OpenAI.


## **Prompt Engineering**

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

We can truncate the input documents in `[DOCUMENTS]` in order to reduce the number of tokens that we have in our input prompt. To do so, all text generation modules have two parameters that we can tweak:

* `doc_length`
    * The maximum length of each document. If a document is longer, it will be truncated. If None, the entire document is passed.
* `tokenizer`
    * The tokenizer used to calculate to split the document into segments used to count the length of a document.
        * If tokenizer is  `'char'`, then the document is split up into characters which are counted to adhere to `doc_length`
        * If tokenizer is `'whitespace'`, the document is split up into words separated by whitespaces. These words are counted       and truncated depending on `doc_length`
        * If tokenizer is `'vectorizer'`, then the internal CountVectorizer is used to tokenize the document. These tokens are counted and truncated depending on `doc_length`
        * If tokenizer is a callable, then that callable is used to tokenized the document. These tokens are counted and truncated depending on `doc_length`

This means that the definition of `doc_length` changes depending on what constitutes a token in the `tokenizer` parameter. If a token is a character, then `doc_length` refers to max length in characters. If a token is a word, then `doc_length` refers to the max length in words.

Let's illustrate this with an example. In the code below, we will use [`tiktoken`](https://github.com/openai/tiktoken) to count the number of tokens in each document and limit them to 100 tokens. All documents that have more than 100 tokens will be truncated.

We start by installing the relevant packages:

```bash
pip install tiktoken openai
```

Then, we use `bertopic.representation.OpenAI` to represent our topics with nicely written labels. We specify that documents that we put in the prompt cannot exceed 100 tokens each. Since we will put 4 documents in the prompt, they will total roughly 400 tokens:

```python
import openai
import tiktoken
from bertopic.representation import OpenAI
from bertopic import BERTopic

# Tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")

# Create your representation model
client = openai.OpenAI(api_key="sk-...")
representation_model = OpenAI(
    client,
    model="gpt-6-luna",
    delay_in_seconds=2,
    nr_docs=4,
    doc_length=100,
    tokenizer=tokenizer
)

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

## **🤗 Transformers**

Nearly every week, there are new and improved models released on the 🤗 [Model Hub](https://huggingface.co/models) that, with some creativity, allow for
further fine-tuning of our c-TF-IDF based topics. These models range from text generation to zero-classification. In BERTopic, wrappers around these
methods are created as a way to support whatever might be released in the future.

Using a model from the huggingface hub, such as Gemma 4, is rather straightforward:

```python
from bertopic.representation import TextGeneration
from bertopic import BERTopic

# Create your representation model
representation_model = TextGeneration('google/gemma-4-E2B-it')

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

You can also pass your own pipeline and prompt:

```python
from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = "I have a topic described by the following keywords: [KEYWORDS]. Give a short label for this topic, and nothing else."

# Create your representation model
generator = pipeline('text-generation', model='google/gemma-4-E2B-it')
representation_model = TextGeneration(generator, prompt=prompt)
```

<br>
<div class="svg_image">
--8<-- "docs/getting_started/representation/hf.svg"
</div>
<br>

A model with a chat template, as instruction-tuned models have, is sent the prompt as a chat message, so you write
the prompt as plain text and its template is applied for you. In a custom prompt, the `[KEYWORDS]` and `[DOCUMENTS]`
tags decide where the keywords and documents are inserted.

## **llama.cpp**

An amazing framework for using LLMs for inference is [`llama.cpp`](https://github.com/ggerganov/llama.cpp) which has [python bindings](https://github.com/abetlen/llama-cpp-python) that we can use in BERTopic. To start with, we first need to install `llama-cpp-python`:

```bash
pip install llama-cpp-python
```

or using the following for hardware acceleration:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

!!! Note
    There are a number of [installation options](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration) depending on your hardware and OS. Make sure that you select the correct one to optimize your performance.

After installation, you need to download your LLM locally before we use it in BERTopic, like so:

```bash
wget https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf
```

Finally, we can now use the model with BERTopic in just a couple of lines:

```python
from bertopic import BERTopic
from bertopic.representation import LlamaCPP

# Use llama.cpp to load in a 4-bit quantized version of Gemma 4 E2B
representation_model = LlamaCPP("gemma-4-E2B-it-Q4_0.gguf")

# Create our BERTopic model
topic_model = BERTopic(representation_model=representation_model,  verbose=True)
```

If you want to have more control over the LLMs parameters, you can run it like so:

```python
from bertopic import BERTopic
from bertopic.representation import LlamaCPP
from llama_cpp import Llama

# Use llama.cpp to load in a 4-bit quantized version of Gemma 4 E2B
llm = Llama(model_path="gemma-4-E2B-it-Q4_0.gguf", n_gpu_layers=-1, n_ctx=16384)
representation_model = LlamaCPP(llm)

# Create our BERTopic model
topic_model = BERTopic(representation_model=representation_model,  verbose=True)
```

The model is prompted through its own chat template, which llama.cpp reads from the GGUF file.

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
client = openai.OpenAI(api_key="sk-...")
representation_model = OpenAI(client)

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
representation_model = OpenAI(client, prompt=prompt)
```

### **Choosing a model**

To choose a specific model from OpenAI's offering:

```python
representation_model = OpenAI(client, model="gpt-6.1-sol", delay_in_seconds=10)
```

Prompting with their models is very satisfying and is customizable as follows:

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

representation_model = OpenAI(client, model="gpt-6-luna", prompt=summarization_prompt, nr_docs=5, delay_in_seconds=3)
```

The above is not constrained to just creating a short description or summary of the topic, we can extract labels, keywords, poems, example documents, extensitive descriptions, and more using this method!
If you want to have multiple representations of a single topic, it might be worthwhile to also check out [**multi-aspect**](https://maartengr.github.io/BERTopic/getting_started/multiaspect/multiaspect.html) topic modeling with BERTopic.

## **Ollama**

To use [Ollama](https://github.com/ollama/ollama) within BERTopic, it is advised to use the `openai` package as it allows to pass through a model using the url on which the model is running.

You will first need to install `openai`:

```bash
pip install openai
```

After installation, usage is straightforward and you can select any model that you have prepared in your `ollama` model list. You can see all models by running `ollama list`.

Select one from the list and you can use it in BERTopic as follows:

```python
import openai
from bertopic.representation import OpenAI
from bertopic import BERTopic

client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1', #wherever ollama is running
    api_key='ollama', # required, but unused
)


# Create your representation model
representation_model = OpenAI(client, model='gemma4')

# Create your BERTopic model
topic_model = BERTopic(representation_model=representation_model,  verbose=True)
```

## **LiteLLM**

An amazing framework to simplify connecting to external LLMs, is [LiteLLM](https://docs.litellm.ai). This package allows you to connect to OpenAI, Cohere, Anthropic, etc. all within one package. This makes iteration and testing out different models a breeze!

o start with, we first need to install `litellm`:

```bash
pip install litellm
```

After installation, usage is straightforward and you can select any model found in their [docs](https://docs.litellm.ai/docs/providers).
Let's show an example with OpenAI:

```python
import os
from bertopic import BERTopic
from bertopic.representation import LiteLLM

# set ENV variables
os.environ["OPENAI_API_KEY"] = "MY_KEY"

# Create your representation model
representation_model = LiteLLM(model="gpt-6-luna")

# Create our BERTopic model
topic_model = BERTopic(representation_model=representation_model,  verbose=True)
```

## **LangChain**

[LangChain](https://github.com/langchain-ai/langchain) is a framework for building applications with large language
models, and it offers one interface to the chat models of many providers. In BERTopic, any LangChain chat model can
label the topics.

To use LangChain, you will need to install the langchain package first, together with the package of the provider you
want to use, like openai:

```bash
pip install langchain langchain-openai
```

Then, you can create your chat model and pass it to BERTopic as follows:

```python
from langchain.chat_models import init_chat_model
from bertopic.representation import LangChain
from bertopic import BERTopic

# Create your representation model
model = init_chat_model("gpt-6-luna", model_provider="openai")
representation_model = LangChain(model)

# Use the representation model in BERTopic on top of the default pipeline
topic_model = BERTopic(representation_model=representation_model)
```

You can also use a custom prompt, in which the `[DOCUMENTS]` and `[KEYWORDS]` tags decide where the representative
documents and keywords are inserted:

```python
prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
Based on the above information, give a short label for the topic, and nothing else.
"""
representation_model = LangChain(model, prompt=prompt)
```

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
co = cohere.ClientV2(my_api_key)
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
