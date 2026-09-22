from __future__ import annotations

import base64
import io
from html import escape
from typing import TYPE_CHECKING

from bertopic._corpus import Modality
from bertopic.plotting._utils import select_topics

if TYPE_CHECKING:
    from IPython.display import HTML
    from PIL import Image

    from bertopic import BERTopic


# Each summary under the name it goes by, in the order a row shows them: the pictures, then the player
SUMMARIES = {Modality.IMAGE: "Collage", Modality.VIDEO: "Frame sheet", Modality.AUDIO: "Montage"}

# Lines and shading are grey at partial opacity, so the table reads the same in light and dark notebooks.
# Its rules name both the wrapper and the table, which outranks the style JupyterLab gives every table
PAGE = """<meta charset="utf-8">
<style>
.bertopic-media {{
  display: inline-block; max-width: 100%; overflow-x: auto;
  border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 8px;
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.bertopic-media .topics {{
  border-collapse: collapse; border-style: hidden; margin: 0; table-layout: auto; color: inherit; font: inherit;
}}
.bertopic-media .topics th, .bertopic-media .topics td {{
  padding: 10px 14px; border: 1px solid rgba(128, 128, 128, 0.35); text-align: left; vertical-align: middle;
}}
.bertopic-media .topics th {{
  background: rgba(128, 128, 128, 0.12); font-size: 12px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.bertopic-media .topics tbody tr {{ background: transparent; }}
.bertopic-media .topics tbody tr:hover {{ background: rgba(128, 128, 128, 0.07); }}
.bertopic-media .topics .number {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bertopic-media figure {{ display: inline-block; margin: 4px 20px 4px 0; vertical-align: top; }}
.bertopic-media img, .bertopic-media audio {{ display: block; border-radius: 4px; }}
.bertopic-media figcaption {{ margin-top: 6px; font-size: 12px; opacity: 0.7; }}
</style>
<div class="bertopic-media"><table class="topics">
<thead><tr><th class="number">Topic</th><th class="number">Count</th><th>Name</th><th>Media</th></tr></thead>
<tbody>
{rows}
</tbody>
</table></div>"""


def visualize_media(
    topic_model: BERTopic,
    topics: list[int] | None = None,
    top_n_topics: int | None = None,
    height: int = 200,
) -> HTML:
    """Show each topic's media beside its name: its pictures, and a montage to play.

    A topic's media is summarised once per modality when it is represented: its images
    tiled into a collage, the middle frame of each video tiled into a frame sheet, and the
    opening seconds of each clip joined into a montage. This lays those summaries out one
    row per topic, so a topic of photographs, recordings and clips is seen and heard at once.

    Plotly has no mark for sound, so the page is plain HTML with every picture and montage
    embedded in it. Showing it needs IPython, which every notebook has.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to show.
        top_n_topics: Only show the top n most frequent topics.
        height: The height of each picture on the page, in pixels.

    Returns:
        An `IPython.display.HTML`, which a notebook shows inline. Its `data` is the page
        itself, which any browser opens once it is written to a file.

    Examples:
    To see the media of a model fitted on images and their captions:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import MultiModalRepresentation

    topic_model = BERTopic(representation_model={"Media": MultiModalRepresentation()})
    topic_model.fit(captions, images=images)
    topic_model.visualize_media()
    ```

    Or to save the ten largest topics as a page:

    ```python
    page = topic_model.visualize_media(top_n_topics=10)
    with open("media.html", "w", encoding="utf-8") as file:
        file.write(page.data)
    ```
    """
    rows = []
    for topic_id in select_topics(topic_model, topics, top_n_topics):
        topic = topic_model._topics[topic_id]
        summaries = topic.media.summaries if topic.media is not None else {}

        # One figure per summary, captioned with what it is, so a collage and a frame sheet are told apart
        figures = [
            f"<figure>{_embed(summaries[modality], height)}<figcaption>{name}</figcaption></figure>"
            for modality, name in SUMMARIES.items()
            if modality in summaries
        ]
        rows.append(
            f'<tr><td class="number">{topic.id}</td><td class="number">{topic.nr_documents}</td>'
            f"<td>{escape(topic.name)}</td><td>{''.join(figures)}</td></tr>"
        )

    # Imported here because IPython comes with every notebook, but BERTopic does not depend on it
    from IPython.display import HTML

    return HTML(PAGE.format(rows="\n".join(rows)))


def _embed(media: Image.Image | bytes, height: int) -> str:
    """Embed media in a page: WAV bytes as a player, or a picture as a JPEG."""
    if isinstance(media, bytes):
        return f'<audio controls src="data:audio/wav;base64,{base64.b64encode(media).decode()}"></audio>'

    # Shrunk to `height` pixels first, so a 600-pixel collage does not bloat the page
    thumbnail = media.copy()
    thumbnail.thumbnail((thumbnail.width, height))
    buffer = io.BytesIO()
    thumbnail.convert("RGB").save(buffer, "JPEG")
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}">'
