Documents or text are often accompanied by imagery or the other way around. For example, social media images with captions and products with descriptions. Topic modeling has traditionally focused on creating topics from textual representations. However, as more multimodal representations are created, the need for multimodal topics increases.

BERTopic can perform **multimodal topic modeling** on text, images, audio, video and code. Each kind is passed to `.fit` or `.fit_transform` under its own name (`documents`, `images=`, `audio=`, `video=` and `code=`), embedded, and clustered just like documents. c-TF-IDF can only count words, so a sample of each topic's media is described in words, and those words become the topic's keywords.

Images need `pip install bertopic[vision]`, audio `bertopic[audio]`, and video `bertopic[video]`, which decodes frames with `torchcodec`.

## **Models**

Two models make this work. The first is an **embedding model** that places your media in a vector space. `MultiModalBackend` wraps a sentence-transformers model, and a model that handles every kind of input lets you cluster any of them:

```python
from sentence_transformers import SentenceTransformer
from bertopic.backend import MultiModalBackend

# One model for text, images, audio and video
model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-omni-nano",
    trust_remote_code=True,
    model_kwargs={"default_task": "retrieval"},
    default_prompt_name="document",
)
embedding_model = MultiModalBackend(model)
```

For `jina-embeddings-v5-omni-nano`, the `retrieval` setting with the `document` prompt is the one that keeps media apart; under `clustering`, audio clips all end up with nearly the same embedding. It needs `transformers>=5`. For images and text alone, CLIP works too:

```python
embedding_model = MultiModalBackend("clip-ViT-B-32")
```

Each kind can also be given a model of its own (`image_model=`, `audio_model=`, `video_model=` and `code_model=`), but rows of different kinds are only comparable when one model embeds them all, so a model per kind suits a corpus of one kind.

The second is a **representation model**. `MultiModalRepresentation` describes up to nine items of each kind per topic with the model for that kind, and is passed as an additional aspect:

```python
from bertopic.representation import MultiModalRepresentation

representation_model = {
    "Media": MultiModalRepresentation(
        "HuggingFaceTB/SmolVLM-256M-Instruct",        # describes images and video frames
        audio_model="openai/whisper-large-v3-turbo",  # transcribes audio
    )
}
```

Each model is loaded the first time a topic needs it. Instead of a model's name, you can also pass any callable that takes a list of items and returns one description each, such as a call to an API. Items that already have text, such as captioned images, keep it.

## **Text + Images**

The most basic example of multimodal topic modeling in BERTopic is when you have images that accompany your documents. This means that it is expected that each document has an image and vice versa. Instagram pictures, for example, almost always have some descriptions to them.

<figure markdown>
  ![Image title](images_and_text.svg)
  <figcaption></figcaption>
</figure>

In this example, we are going to use images from `flickr` that each have a caption associated to it:

```python
# NOTE: This requires the `datasets` package which you can
# install with `pip install datasets`
from datasets import load_dataset

ds = load_dataset("maderix/flickr_bw_rgb")
images = ds["train"]["image"]
docs = ds["train"]["caption"]
```

The `docs` variable contains the captions for each image in `images`. We can now use these variables to run our multimodal example:

!!! Tip
    Do note that it is better to pass the paths of the images instead of the images themselves as there is no need to keep all images in memory. When passing the paths of the images, they are only opened temporarily when they are needed.

```python
from bertopic import BERTopic
from bertopic.representation import MultiModalRepresentation

# Additional ways of representing a topic
media_model = MultiModalRepresentation()

# Make sure to add the `media_model` to a dictionary
representation_model = {
   "Media":  media_model,
}
topic_model = BERTopic(representation_model=representation_model, verbose=True)
topics, probs = topic_model.fit_transform(docs, images=images)
```

In this example, we are clustering the documents and are then looking for the best matching images to the resulting clusters. Each topic's images are tiled into a collage, which `topic_model.visualize_media()` shows beside its name (see [Visualizing media](#visualizing-media)):

<br><br>
<img src="images_and_text.jpg">
<br><br>

!!! Tip
    In the example above, we are clustering the documents but since you have
    images, you might want to cluster those or cluster an aggregation of both
    images and documents. For that, you can use the new `MultiModalBackend`
    to generate embeddings:

    ```python
    import numpy as np
    from bertopic.backend import MultiModalBackend
    model = MultiModalBackend('clip-ViT-B-32', batch_size=32)

    # Embed documents only
    doc_embeddings = model.embed_documents(docs)

    # Embedding images only
    image_embeddings = model.embed_media(images, "image")

    # Average both, which is what passing documents and images together does
    doc_image_embeddings = np.mean([doc_embeddings, image_embeddings], axis=0)
    ```

## **Images Only**

Traditional topic modeling techniques can only be run on textual data, as is shown in the example above. However, there are plenty of cases where textual data is not available but images are. BERTopic allows topic modeling to be performed using only images as your input data.

<figure markdown>
  ![Image title](images_only.svg)
  <figcaption></figcaption>
</figure>

To run BERTopic on images only, we first need to embed our images and then define a model that convert images to text. To do so, we are going to need some images. We will take the same images as the above but instead save them locally and pass the paths to the images instead. As mentioned before, this will make sure that we do not hold too many images in memory whilst only a small subset is needed:


```python
import os
import glob
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import util

# Flickr 8k images
img_folder = 'photos/'
caps_folder = 'captions/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)

    if not os.path.exists('Flickr8k_Dataset.zip'):   #Download dataset if does not exist
        util.http_get('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', 'Flickr8k_Dataset.zip')
        util.http_get('https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', 'Flickr8k_text.zip')

    for folder, file in [(img_folder, 'Flickr8k_Dataset.zip'), (caps_folder, 'Flickr8k_text.zip')]:
        with zipfile.ZipFile(file, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                zf.extract(member, folder)
images = list(glob.glob('photos/Flicker8k_Dataset/*.jpg'))
```

Next, we can run our pipeline:


```python
from bertopic.representation import KeyBERTInspired, MultiModalRepresentation
from bertopic.backend import MultiModalBackend

# Image embedding model
embedding_model = MultiModalBackend('clip-ViT-B-32', batch_size=32)

# Describing the images is what gives an image-only corpus its keywords
representation_model = {
    "Media": MultiModalRepresentation("HuggingFaceTB/SmolVLM-256M-Instruct")
}

```

Using these models, we can run our pipeline:

```python
from bertopic import BERTopic

# Train our model with images only
topic_model = BERTopic(embedding_model=embedding_model, representation_model=representation_model, min_topic_size=30)
topics, probs = topic_model.fit_transform(documents=None, images=images)
```

The descriptions become each topic's keywords, and `topic_model.get_representation(topic, "Media").captions` holds what was written about a topic's images:

<br><br>
<img src="images_only.jpg">
<br><br>

!!! Tip
    A vision model tends to start every description the same way, such as "In this image we can see", and those words would then top every topic's keywords. Passing `vectorizer_model=CountVectorizer(stop_words="english")` to BERTopic leaves them out.

When text and images share one space, as they do with CLIP, image topics can be searched with a sentence: `topic_model.find_topics("dogs playing in the snow")`.

## **Audio**

This section and the ones after it use the jina embedding model and the describing model from [Models](#models). Audio is passed as file paths or as arrays. An array is taken to be sampled at 16 kHz, which is what Whisper expects. Here, 563 recordings of people calling their bank:

```python
import io
import librosa
from datasets import Audio, load_dataset

# `decode=False` hands over each recording's bytes, which librosa reads at 16 kHz
minds = load_dataset("PolyAI/minds14", "en-US", split="train").cast_column("audio", Audio(decode=False))
clips = [librosa.load(io.BytesIO(row["bytes"]), sr=16_000)[0] for row in minds["audio"]]

# Representation model
representation_model = {
    "Media": MultiModalRepresentation(
        audio_model="openai/whisper-large-v3-turbo",  # transcribes audio
    )
}

# Fit BERTopic
topic_model = BERTopic(embedding_model=embedding_model, representation_model=representation_model)
topics, probs = topic_model.fit_transform(audio=clips)
```

Whisper transcribes nine calls from each topic, and their transcripts become its keywords. A call longer than Whisper's 30 seconds is transcribed in 30-second windows. Transcribing audio given as file paths needs [ffmpeg](https://ffmpeg.org/) installed, which arrays do not.

## **Video**

Videos are passed as file paths:

```python
from pathlib import Path

videos = [str(path) for path in Path("videos").glob("*.mp4")]

# Representation model
representation_model = {
    "Media": MultiModalRepresentation(
        "HuggingFaceTB/SmolVLM-256M-Instruct",        # describes images and video frames
    )
}

# Fit BERTopic
topic_model = BERTopic(embedding_model=embedding_model, representation_model=representation_model)
topics, probs = topic_model.fit_transform(video=videos)
```

The vision model describes three frames of each representative clip (`nr_frames`), taken from the middle of equal stretches of the clip so that black openings and endings are skipped, and the descriptions of a clip's frames are joined into one. A topic needs ten items by default, so for a folder of only a few dozen clips, lower `min_topic_size`.

## **Code**

Code is its own kind of input. It is embedded by the code model when the backend has one (`code_model=`), and c-TF-IDF reads it as it is, so it needs no describing model. For example, every function in a project:

```python
import ast
from pathlib import Path

functions = []
for path in Path("my_project").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    nodes = ast.walk(ast.parse(source))
    functions += [ast.get_source_segment(source, node) for node in nodes if isinstance(node, ast.FunctionDef)]

topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(code=functions)
```

## **Several kinds at once**

Several kinds can be passed together, and are then clustered in one model:

```python
topics, probs = topic_model.fit_transform(images=images, audio=clips, video=videos)
```

How documents are read depends on how many there are. When there are as many documents as media, each document describes the item at its position, as captions do, and each pair becomes one row. With any other number, the documents are rows of their own beside the media. The rows are ordered images, audio, video, documents and then code, which is also the order of `topics`.

Rows of different kinds are only comparable when one model embeds them all. Whether a photograph and a call about the same subject end up in the same topic depends on that model.

## **Visualizing media**

Each topic keeps its representative media, the items chosen to stand for it, and one summary per kind of media: its images tiled into a collage, the middle frame of each clip tiled into a frame sheet, and the first two seconds of each recording joined into a montage.

```python
topic_model.representative_items_     # every topic's representative media, whatever their kind
topic_model.representative_images_    # every topic's collage

media = topic_model.get_representation(topic, "Media")
media.items, media.summaries, media.captions
```

The representative items are also in the `Representative_Items` column of `topic_model.get_topic_info()`. To see each topic's collage, frame sheet and montage beside its name, run:

```python
topic_model.visualize_media()
```

<br><br>
<img src="media_page.jpg">
<br><br>

Each montage adds about 0.8 MB to the page, so for a model with many audio topics, `top_n_topics=10` keeps the page small. The page is IPython's `HTML`, so a notebook shows it inline, and its `data` is a page that any browser opens once saved:

```python
with open("media.html", "w", encoding="utf-8") as file:
    file.write(topic_model.visualize_media().data)
```

The interactive datamap shows every image and video. Pass the same documents and media you passed to `fit`, so that every point lines up with its topic. Hovering over an image then shows it, and hovering over a video shows its middle frame, at about 3 KB of page per picture:

```python
topic_model.visualize_document_datamap(docs, images=images, interactive=True)
```

<br><br>
<img src="datamap_media.jpg">
<br><br>

The datamap needs `pip install bertopic[datamap]`.

## **Saving**

With `serialization="safetensors"` or `"pytorch"`, `topic_model.save` keeps each topic's summaries, with collages and frame sheets as images and montages as audio files. Representative items are kept only as their paths, since the media they point to is already on disk, so pass your media as paths if you want them back after loading. The default, `pickle`, pickles the whole model, media included.
