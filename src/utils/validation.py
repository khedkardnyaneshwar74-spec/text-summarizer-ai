"""
Input validation utilities for TextSummarize AI.
Ensures security, sensible document bounds, and clean error messages.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class InputValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    cleaned_text: str = ""
    word_count: int = 0
    char_count: int = 0


# Default constraints
MIN_WORDS_FOR_SUMMARY = 15
MAX_CHARACTERS_LIMIT = 100_000
MAX_RECOMMENDED_TRANSFORMER_WORDS = 1024


def clean_and_normalize_text(text: str) -> str:
    """Normalize whitespace and strip control characters while preserving sentence punctuation."""
    if not text:
        return ""
    # Normalize carriage returns and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple consecutive blank lines or excessive spaces
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join([line for line in lines if line])
    return cleaned.strip()


def validate_input_text(
    text: Optional[str],
    min_words: int = MIN_WORDS_FOR_SUMMARY,
    max_chars: int = MAX_CHARACTERS_LIMIT,
) -> InputValidationResult:
    """
    Validate and clean input text for summarization.
    """
    if text is None or not text.strip():
        return InputValidationResult(
            is_valid=False,
            error_message="Input text cannot be empty. Please enter text or upload a .txt document.",
        )

    cleaned = clean_and_normalize_text(text)
    words = cleaned.split()
    word_count = len(words)
    char_count = len(cleaned)

    if char_count > max_chars:
        return InputValidationResult(
            is_valid=False,
            error_message=f"Document exceeds maximum supported size ({char_count:,} characters > {max_chars:,} limit). Please shorten your document.",
            cleaned_text=cleaned,
            word_count=word_count,
            char_count=char_count,
        )

    if word_count < min_words:
        return InputValidationResult(
            is_valid=False,
            error_message=f"Input document is too short ({word_count} words). A meaningful summary requires at least {min_words} words.",
            cleaned_text=cleaned,
            word_count=word_count,
            char_count=char_count,
        )

    warning = None
    if word_count > MAX_RECOMMENDED_TRANSFORMER_WORDS:
        warning = (
            f"Input contains {word_count:,} words, which exceeds standard Transformer context windows (~1024 tokens). "
            "The system will automatically apply intelligent chunk-and-combine or sequence truncation."
        )

    return InputValidationResult(
        is_valid=True,
        error_message=None,
        warning_message=warning,
        cleaned_text=cleaned,
        word_count=word_count,
        char_count=char_count,
    )


def validate_uploaded_file(file_name: str, file_bytes: bytes) -> Tuple[bool, Optional[str], str]:
    """
    Validate uploaded file format and decode text.
    Returns: (is_valid, error_message, decoded_text)
    """
    if not file_name.lower().endswith(".txt"):
        return False, f"Unsupported file type for '{file_name}'. Only plain text (.txt) files are supported.", ""

    # Attempt decodings safely
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            decoded = file_bytes.decode(encoding)
            return True, None, decoded
        except (UnicodeDecodeError, AttributeError):
            continue

    return False, "Could not decode file content. Please ensure the file is valid UTF-8 plain text.", ""
