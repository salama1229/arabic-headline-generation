"""Arabic text preprocessing, copied 1:1 from the project notebook (NLP2.ipynb).

Keeping these functions identical to the notebook is what guarantees that the
Streamlit app feeds the models exactly the same token ids that they saw in training.
"""
from __future__ import annotations

import html
import re

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"

MAX_ARTICLE_LEN = 80   # words kept from the article (encoder input)
MAX_TITLE_LEN = 20     # headline length incl. [SOS]/[EOS]


def clean_arabic(text: str) -> str:
    """Normalise an Arabic string (HTML entities, URLs, tags, diacritics, alef forms)."""
    text = html.unescape(text)                                   # &quot; -> "
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)           # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                         # remove HTML tags
    text = re.sub(r"[ؐ-ًؚ-ٰٟۖ-ۭ]", "", text)  # Arabic marks
    text = text.replace("ـ", "")                            # remove Tatweel
    text = re.sub("[أإآٱ]", "ا", text)  # unify alef forms
    text = re.sub(r"\s+", " ", text).strip()                     # collapse spaces
    return text


def tokenize_text(text: str) -> list[str]:
    """Whitespace tokenisation (punctuation stays attached to words, as in the notebook)."""
    return text.split()
