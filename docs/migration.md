---
hide:
  - navigation
---

# Migrating to 1.0

BERTopic 1.0 rewrote most of what happens under the hood, while keeping its API as it was wherever possible. Most code written for 0.17.4 runs as it is, but some of it needs a small update. Below, you can find each of those changes with the code for 0.17.4 and for 1.0. For everything else that changed, see the [changelog](changelog.md).

In the examples, `topic_model` is a fitted model and `docs` the documents it was fitted on.

## **Images**

`VisualRepresentation` is replaced by `MultiModalRepresentation`, which describes images, audio and video, each with a model of its own. CLIP still works, but you can now embed the images with EmbeddingGemma 2, which embeds text, images, audio and video in one space:

=== "0.17.4"

    ```python
    from bertopic import BERTopic
    from bertopic.backend import MultiModalBackend
    from bertopic.representation import VisualRepresentation

    embedding_model = MultiModalBackend("clip-ViT-B-32", batch_size=32)
    representation_model = {
        "Visual_Aspect": VisualRepresentation(image_to_text_model="nlpconnect/vit-gpt2-image-captioning")
    }
    topic_model = BERTopic(embedding_model=embedding_model, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(documents=None, images=images)
    ```

===+ "1.0"

    ```python
    from bertopic import BERTopic
    from bertopic.representation import MultiModalRepresentation

    representation_model = {"Media": MultiModalRepresentation(image_model="HuggingFaceTB/SmolVLM-256M-Instruct")}
    topic_model = BERTopic(embedding_model="google/embeddinggemma-2", representation_model=representation_model)
    topics, probs = topic_model.fit_transform(images=images)
    ```

Each topic's collage, which 0.17.4 kept in `topic_aspects_["Visual_Aspect"]`, is now in `topic_model.representative_images_`. In `.get_topic_info()`, the images themselves are in a `Representative_Items` column instead of a column named after the aspect, and the aspect, `topic_model.topic_aspects_["Media"]`, holds the captions. See the [multimodal guide](getting_started/multimodal/multimodal.md) for audio, video and code.

## **Changing `topics_`**

`topics_`, `probabilities_`, `topic_sizes_` and the other fitted attributes can no longer be overwritten, since BERTopic keeps them in line with each other. To change the topics, for example after reducing outliers, pass them to `.update_topics`:

=== "0.17.4"

    ```python
    new_topics = topic_model.reduce_outliers(docs, topic_model.topics_)
    topic_model.topics_ = new_topics
    ```

===+ "1.0"

    ```python
    new_topics = topic_model.reduce_outliers(docs, topic_model.topics_)
    topic_model.update_topics(docs, topics=new_topics)
    ```

In online topic modeling, `.partial_fit` now keeps track of the topics of every batch itself, so you no longer need to collect them:

=== "0.17.4"

    ```python
    topics = []
    for chunk in doc_chunks:
        topic_model.partial_fit(chunk)
        topics.extend(topic_model.topics_)

    topic_model.topics_ = topics
    ```

===+ "1.0"

    ```python
    for chunk in doc_chunks:
        topic_model.partial_fit(chunk)
    ```

## **`.update_topics` keeps your models**

In 0.17.4, `.update_topics` went back to the default vectorizer, c-TF-IDF and representation models unless you passed yours again, so stop words returned after reducing outliers. In 1.0, it keeps the models of your topic model:

=== "0.17.4"

    ```python
    topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
    ```

===+ "1.0"

    ```python
    topic_model.update_topics(docs, topics=new_topics)
    ```

If you relied on the reset, set the default models yourself before updating:

```python
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

topic_model.vectorizer_model = CountVectorizer(ngram_range=topic_model.n_gram_range)
topic_model.ctfidf_model = ClassTfidfTransformer()
topic_model.representation_model = None
topic_model.update_topics(docs, topics=new_topics)
```

## **Probabilities**

With `calculate_probabilities=True`, `probabilities_`, and the probabilities from `.transform`, have a column for every topic, including the outlier topic when there is one, so each row sums to 1. In 0.17.4, column 3 was topic 3, but with outliers, column 3 is now topic 2. Look the column up by its topic instead:

=== "0.17.4"

    ```python
    probabilities = topic_model.probabilities_[:, 3]
    ```

===+ "1.0"

    ```python
    column = topic_model.topics.topic_ids().index(3)
    probabilities = topic_model.probabilities_[:, column]
    ```

## **Representation models over time, per class and in hierarchies**

`.topics_over_time`, `.topics_per_class` and `.hierarchical_topics` show c-TF-IDF keywords, since running a representation model, such as an LLM, for every timestamp, class or merged topic can be costly. To run your representation model, as 0.17.4 did, pass `use_representation_model=True`:

=== "0.17.4"

    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    ```

===+ "1.0"

    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps, use_representation_model=True)
    ```

## **Zero-shot topics and `nr_topics`**

`nr_topics` and `.reduce_topics` no longer merge zero-shot topics, so they keep their names. This means that `nr_topics` has to be larger than the number of zero-shot topics plus the outlier topic:

=== "0.17.4"

    ```python
    topic_model = BERTopic(zeroshot_topic_list=["space", "religion", "cars", "hockey"], nr_topics=5)
    ```

===+ "1.0"

    ```python
    topic_model = BERTopic(zeroshot_topic_list=["space", "religion", "cars", "hockey"], nr_topics=6)
    ```

## **LLM representations**

`OpenAI` no longer accepts `chat=True`, which already did nothing in 0.17.4. Its default model is now `gpt-6-luna`, since `gpt-4o-mini` shuts down in January 2027:

=== "0.17.4"

    ```python
    representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)
    ```

===+ "1.0"

    ```python
    representation_model = OpenAI(client)
    ```

`LangChain` takes a chat model instead of a chain, since LangChain removed `load_qa_chain`:

=== "0.17.4"

    ```python
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=my_openai_api_key), chain_type="stuff")
    representation_model = LangChain(chain)
    ```

===+ "1.0"

    ```python
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-6-luna", model_provider="openai")
    representation_model = LangChain(model)
    ```

`Cohere` uses Cohere's v2 API, so it takes a `cohere.ClientV2`:

=== "0.17.4"

    ```python
    representation_model = Cohere(cohere.Client(my_api_key))
    ```

===+ "1.0"

    ```python
    representation_model = Cohere(cohere.ClientV2(my_api_key))
    ```

`LlamaCPP`'s `pipeline_kwargs` is now `generator_kwargs`, as in the other LLM representations:

=== "0.17.4"

    ```python
    representation_model = LlamaCPP(llm, pipeline_kwargs={"max_tokens": 50})
    ```

===+ "1.0"

    ```python
    representation_model = LlamaCPP(llm, generator_kwargs={"max_tokens": 50})
    ```

`TextGeneration` takes `text-generation` pipelines only, since transformers 5 removed `text2text-generation`. A model with a chat template now gets its prompt as a chat message, so if your prompt writes a chat template by hand, as 0.17.4's Zephyr and Llama 2 examples did, remove it:

=== "0.17.4"

    ```python
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    representation_model = TextGeneration(generator)
    ```

===+ "1.0"

    ```python
    generator = pipeline("text-generation", model="google/gemma-4-E2B-it")
    representation_model = TextGeneration(generator)
    ```

## **Custom representation models**

A custom representation model gets the documents as a `corpus` and the keywords of each topic as `topic_representations`, and returns a representation for each topic, such as `Keywords` or `Label`:

=== "0.17.4"

    ```python
    from bertopic.representation._base import BaseRepresentation


    class CustomRepresentationModel(BaseRepresentation):
        def extract_topics(self, topic_model, documents, c_tf_idf, topics):
            updated_topics = topics.copy()
            return updated_topics
    ```

===+ "1.0"

    ```python
    from bertopic.representation._base import BaseRepresentation


    class CustomRepresentationModel(BaseRepresentation):
        def extract_topics(self, topic_model, corpus, topic_representations, c_tf_idf):
            updated_topics = topic_representations.copy()
            return updated_topics
    ```

Where 0.17.4 returned a list of words and scores for each topic, 1.0 returns a representation:

=== "0.17.4"

    ```python
    updated_topics = {1: [("space", 0.9), ("nasa", 0.7)]}
    ```

===+ "1.0"

    ```python
    from bertopic._topics import Keywords, Label

    updated_topics = {1: Keywords([("space", 0.9), ("nasa", 0.7)]), 2: Label("Science articles")}
    ```

## **Custom embedding backends**

`WordDocEmbedder` and `BaseEmbedder`'s `word_embedding_model` are removed. Instead, a custom backend can define `embed_words` for the short texts that BERTopic compares with your documents, such as the search term in `.find_topics`:

=== "0.17.4"

    ```python
    from bertopic.backend import WordDocEmbedder

    embedding_model = WordDocEmbedder(embedding_model=document_model, word_embedding_model=word_model)
    ```

===+ "1.0"

    ```python
    from bertopic.backend import BaseEmbedder


    class CustomEmbedder(BaseEmbedder):
        def embed(self, documents, verbose=False):
            return document_model.encode(documents, show_progress_bar=verbose)

        def embed_words(self, words, verbose=False):
            return word_model.encode(words, show_progress_bar=verbose)


    embedding_model = CustomEmbedder()
    ```

`MultiModalBackend.embed_images` is replaced by `.embed_media`, which takes any kind of media, and `.embed` no longer takes images:

=== "0.17.4"

    ```python
    image_embeddings = embedding_model.embed_images(images, verbose=False)
    ```

===+ "1.0"

    ```python
    image_embeddings = embedding_model.embed_media(images, "image")
    ```

## **`TopicMapper`**

`TopicMapper`, `topic_mapper_` and `mappings_` are removed. To find the topic that an original cluster belongs to after merging or reducing topics, use the mapping of `topic_model.topics`:

=== "0.17.4"

    ```python
    topic = topic_model.topic_mapper_.get_mappings()[cluster]
    ```

===+ "1.0"

    ```python
    topic = topic_model.topics.mapping.map(cluster)
    ```

## **The embedding model**

The default embedding model is now EmbeddingGemma 2 for every language. To keep the model of 0.17.4, pass `"sentence-transformers/all-MiniLM-L6-v2"` for English, or `"paraphrase-multilingual-MiniLM-L12-v2"` for other languages:

```python
topic_model = BERTopic(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
```

An `embedding_model` that BERTopic does not recognise, such as a lambda or a model loaded with `transformers`, now raises a `TypeError`. 0.17.4 replaced it with `all-MiniLM-L6-v2` without a word. Pass a model name or a `SentenceTransformer` instead, or wrap your model in a [custom backend](#custom-embedding-backends).

## **Saved models**

Models saved with `safetensors` or `pytorch` by 0.17.4 or earlier load as before. A pickled model, which `.save` gives by default, only loads in the version that saved it, so load it with 0.17.4 and save it again with `safetensors`:

```python
topic_model = BERTopic.load("my_model")
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("my_model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
```

Then, load it with 1.0:

```python
topic_model = BERTopic.load("my_model_dir")
```

A model saved with `safetensors` keeps its topics but not its UMAP and HDBSCAN models, so `.transform` assigns documents by their similarity to the topic embeddings. To keep those models, fit the model again with 1.0.

If a saved model points to an embedding model with a `Clustering` prompt, such as `google/embeddinggemma-300m`, BERTopic now embeds new documents with that prompt. To embed them as 0.17.4 did, pass the model yourself:

```python
from sentence_transformers import SentenceTransformer

topic_model = BERTopic.load("my_model", embedding_model=SentenceTransformer("google/embeddinggemma-300m"))
```

## **Python and dependencies**

BERTopic 1.0 needs Python 3.11 or newer, so on Python 3.10, pip installs 0.17.4. It also needs `sentence-transformers>=6.1` and `transformers[vision]>=5.19`, which installs Pillow and torchvision, since EmbeddingGemma 2 needs them even for text.
