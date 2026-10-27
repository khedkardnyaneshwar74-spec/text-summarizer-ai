"""
Text statistics and analytics utilities for TextSummarize AI.
Calculates word count, sentence count, reading time, and exact compression ratio.
"""

from typing import Dict, Any, List
import re


def count_words(text: str) -> int:
    """Return word count using whitespace splitting."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Return character count."""
    if not text:
        return 0
    if include_spaces:
        return len(text)
    return len(re.sub(r"\s+", "", text))


def approximate_sentences(text: str) -> List[str]:
    """Fallback sentence splitter without external library dependencies."""
    if not text or not text.strip():
        return []
    # Match sentences ending with . ! ? followed by space or newline or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def compute_compression_ratio(original_word_count: int, summary_word_count: int) -> float:
    """
    Compression Ratio (%) = (1 - Summary Word Count / Original Word Count) * 100
    Higher ratio = more concise / condensed summary.
    """
    if original_word_count <= 0:
        return 0.0
    if summary_word_count >= original_word_count:
        return 0.0
    ratio = (1.0 - (summary_word_count / float(original_word_count))) * 100.0
    return round(max(0.0, min(100.0, ratio)), 2)


def estimate_reading_time_seconds(word_count: int, wpm: int = 200) -> float:
    """Estimate reading time in seconds based on 200 words per minute average."""
    if word_count <= 0:
        return 0.0
    return round((word_count / wpm) * 60.0, 1)


def compute_text_statistics(text: str, label: str = "Document") -> Dict[str, Any]:
    """
    Compute comprehensive text statistics for a given text snippet.
    """
    cleaned = text.strip() if text else ""
    words = cleaned.split() if cleaned else []
    sentences = approximate_sentences(cleaned)
    unique_words = set(w.lower() for w in words)

    word_count = len(words)
    char_count = len(cleaned)
    sentence_count = len(sentences)
    avg_sentence_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0.0
    vocab_diversity = round((len(unique_words) / word_count) * 100, 1) if word_count > 0 else 0.0

    return {
        "label": label,
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length_words": avg_sentence_len,
        "unique_words": len(unique_words),
        "vocabulary_diversity_pct": vocab_diversity,
        "reading_time_sec": estimate_reading_time_seconds(word_count),
    }
