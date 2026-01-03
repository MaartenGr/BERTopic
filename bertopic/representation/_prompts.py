DEFAULT_COMPLETION_PROMPT = "What are these documents about? Please give a single label."
DEFAULT_CHAT_PROMPT = """You will extract a short topic label from given documents and keywords.
Here are two examples of topics you created before:

# Example 1
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
topic: Environmental impacts of eating meat

# Example 2
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
topic: Shipping and delivery issues

# Your task
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]

Based on the information above, extract a short topic label (three words at most) in the following format:
topic: <topic_label>
"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts."
DEFAULT_JSON_SCHEMA = {
    "properties": {
        "topic_label": {
            "description": "A short, human-readable label that summarizes the main topic.",
            "title": "Topic Label",
            "type": "string",
        }
    },
    "required": ["topic_label"],
    "title": "Topic",
    "type": "object",
}
