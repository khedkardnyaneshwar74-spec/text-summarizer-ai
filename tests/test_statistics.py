"""
Unit tests for text statistics and compression ratio calculation.
"""

import pytest
from src.utils.text_statistics import (
    count_words,
    count_characters,
    compute_compression_ratio,
    approximate_sentences,
    compute_text_statistics,
)


def test_count_words():
    assert count_words("") == 0
    assert count_words("   ") == 0
    assert count_words("Five words are in here") == 5


def test_count_characters():
    assert count_characters("hello") == 5
    assert count_characters("hello world", include_spaces=True) == 11
    assert count_characters("hello world", include_spaces=False) == 10


def test_compute_compression_ratio():
    # If summary is 25 words from 100 words -> 75% compression
    assert compute_compression_ratio(100, 25) == 75.0
    # If summary is 50 words from 100 words -> 50% compression
    assert compute_compression_ratio(100, 50) == 50.0
    # If summary is equal to original -> 0% compression
    assert compute_compression_ratio(100, 100) == 0.0
    # Edge cases
    assert compute_compression_ratio(0, 50) == 0.0
    assert compute_compression_ratio(100, 120) == 0.0


def test_compute_text_statistics():
    text = "First sentence here. Second sentence follows. Third sentence wraps up."
    stats = compute_text_statistics(text, label="TestDoc")
    assert stats["word_count"] > 0
    assert stats["sentence_count"] == 3
    assert stats["char_count"] == len(text)
    assert stats["reading_time_sec"] > 0
    assert stats["vocabulary_diversity_pct"] > 0
