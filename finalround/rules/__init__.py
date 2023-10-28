import json
from typing import Text


def adjust_intent(text: Text, intent: Text):
    adjusted_intent = intent
    if "%" in text and "mức độ" in intent:
        adjusted_intent = intent.replace("mức độ", "độ sáng")
    return adjusted_intent
