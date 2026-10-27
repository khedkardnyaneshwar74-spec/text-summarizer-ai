"""
Unit tests for Extractive Summarization (TF-IDF Sentence Ranking).
"""

import pytest
from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.preprocessing import split_into_sentences


def test_split_into_sentences():
    text = "The sun rose over the horizon. A cool breeze swept across the valley! Did you see the eagle?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert "The sun rose over the horizon." in sentences[0]


def test_extractive_summarize_preserves_order():
    """
    Extractive summarization must restore chronological sentence order
    to maintain coherent document flow.
    """
    text = (
        "Sentence Alpha establishes the general domain context. "
        "Sentence Beta explains secondary background details that are less salient. "
        "Sentence Gamma introduces the primary revolutionary breakthrough in computing. "
        "Sentence Delta discusses potential future implications for society."
    )
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(text, num_sentences=2)

    assert res["summary"] != ""
    assert len(res["selected_sentences"]) == 2
    # Verify chronological sequence in resulting summary
    idx1 = text.find(res["selected_sentences"][0])
    idx2 = text.find(res["selected_sentences"][1])
    assert idx1 < idx2, "Selected sentences must be ordered as they originally appeared in the text."


def test_extractive_empty_text():
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize("")
    assert res["summary"] == ""
    assert res["total_sentences"] == 0
    assert len(res["selected_sentences"]) == 0


def test_extractive_fewer_sentences_than_requested():
    text = "Only one sentence is present here."
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(text, num_sentences=3)
    assert res["summary"] == "Only one sentence is present here."
    assert res["total_sentences"] == 1
