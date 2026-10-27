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

    try:
        import nltk
        sentences = nltk.sent_tokenize(normalized)
    except (ImportError, LookupError, Exception):
        sentences = []

    if not sentences:
        # Robust regex-based sentence segmentation fallback
        # Handles standard sentence terminations, quotes, brackets, and uppercase follow-up
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])"
        raw_parts = re.split(pattern, normalized)
        for part in raw_parts:
            # Further split on newline breaks if they separate sentences
            lines = part.split("\n")
            for line in lines:
                cleaned_line = line.strip()
                if cleaned_line:
                    sentences.append(cleaned_line)

    # Filter out empty or whitespace-only sentences
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    return cleaned_sentences if cleaned_sentences else [text.strip()]


def preprocess_document(text: str) -> str:
    """Preprocess document before passing to summarization engines."""
    if not text:
        return ""
    # Strip unnecessary spaces, normalize linebreaks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)
