"""
Unit tests for input validation and file decoding.
"""

import pytest
from src.utils.validation import (
    validate_input_text,
    validate_uploaded_file,
    clean_and_normalize_text,
)


def test_validate_empty_input():
    result = validate_input_text("")
    assert not result.is_valid
    assert "cannot be empty" in result.error_message

    none_res = validate_input_text(None)
    assert not none_res.is_valid
    assert "cannot be empty" in none_res.error_message


def test_validate_short_text():
    short_text = "This is a single short sentence."
    result = validate_input_text(short_text, min_words=15)
    assert not result.is_valid
    assert "too short" in result.error_message


def test_validate_valid_text():
    valid_text = (
        "Artificial intelligence and deep learning models have expanded significantly. "
        "Modern natural language processing architectures can synthesize complex documents "
        "into concise, coherent, and readable summaries for researchers and practitioners worldwide."
    )
    result = validate_input_text(valid_text, min_words=15)
    assert result.is_valid
    assert result.error_message is None
    assert result.word_count >= 15


def test_clean_and_normalize_text():
    raw = "  Line 1 with   extra spaces. \r\n\r\n  Line 2 with trailing tabs.\t  "
    cleaned = clean_and_normalize_text(raw)
    assert "\r" not in cleaned
    assert "Line 1 with   extra spaces." in cleaned
    assert "Line 2 with trailing tabs." in cleaned


def test_validate_uploaded_file_valid_txt():
    content = "Sample document content inside uploaded text file.".encode("utf-8")
    is_valid, error, text = validate_uploaded_file("document.txt", content)
    assert is_valid
    assert error is None
    assert "Sample document" in text


def test_validate_uploaded_file_invalid_extension():
    content = b"%PDF-1.4 dummy binary"
    is_valid, error, text = validate_uploaded_file("document.pdf", content)
    assert not is_valid
    assert "Only plain text (.txt) files are supported" in error
