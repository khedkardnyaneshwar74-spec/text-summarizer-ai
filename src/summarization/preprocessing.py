"""
Preprocessing module for TextSummarize AI.
Handles sentence splitting, normalization, and tokenization fallbacks.
"""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split document text into clean, non-empty sentences.
    Attempts NLTK sent_tokenize, and falls back gracefully to high-precision regex.
    """
    if not text or not text.strip():
        return []

    # Clean redundant whitespace first
    normalized = re.sub(r"\r\n|\r", "\n", text)
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()

    sentences: List[str] = []

    # Multilingual sentence splitting supporting English (. ! ?), Hindi/Marathi (। ॥), and newline boundaries
    # If text contains Devanagari characters (U+0900 to U+097F) or danda (। ॥)
    is_indic = bool(re.search(r"[\u0900-\u097F।॥]", normalized))
    
    if is_indic:
        # Split on danda, double danda, period, exclamation, or question mark followed by whitespace or linebreaks
        raw_parts = re.split(r"(?<=[।॥.!?\n])\s*", normalized)
        for part in raw_parts:
            p = part.strip()
            if p:
                sentences.append(p)
    else:
        try:
            import nltk
            sentences = nltk.sent_tokenize(normalized)
        except (ImportError, LookupError, Exception):
            sentences = []

        if not sentences:
            pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])"
            raw_parts = re.split(pattern, normalized)
            for part in raw_parts:
                lines = part.split("\n")
                for line in lines:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        sentences.append(cleaned_line)

    # Filter out empty or degenerate sentences
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    return cleaned_sentences if cleaned_sentences else [text.strip()]


def preprocess_document(text: str) -> str:
    """Preprocess document before passing to summarization engines."""
    if not text:
        return ""
    # Strip unnecessary spaces, normalize linebreaks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)
