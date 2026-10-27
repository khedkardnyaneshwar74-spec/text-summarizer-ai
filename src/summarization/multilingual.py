"""
Multilingual NLP utilities supporting English, Hindi (हिंदी), and Marathi (मराठी).
Provides language detection, Devanagari sentence boundary processing, and Indic stopwords.
"""

import re
from typing import Set, Tuple, List

# Core Hindi Stopwords
HINDI_STOPWORDS: Set[str] = {
    "के", "का", "की", "में", "है", "हैं", "और", "से", "पर", "को", "था", "थे", "थी",
    "यह", "वह", "ने", "एक", "भी", "इस", "कि", "जो", "तो", "ही", "या", "हो", "गया",
    "करने", "किया", "जाता", "दिया", "हुए", "रहे", "रहा", "रही", "होने", "अपने", "उन्हें",
    "उसके", "सकते", "सकता", "सकती", "हुआ", "हुई", "जब", "तब", "अगर", "तक", "अन्य",
}

# Core Marathi Stopwords
MARATHI_STOPWORDS: Set[str] = {
    "आणि", "आहे", "आहेत", "या", "त्या", "च्या", "ना", "नी", "वर", "मध्ये", "होता", "होती",
    "होते", "एक", "करणे", "केले", "झाले", "झाला", "म्हणून", "पण", "की", "नाही", "हे", "तो",
    "ती", "ते", "असे", "तसे", "केली", "झाली", "त्यांनी", "त्यांना", "सर्व", "काही", "पुन्हा",
    "तसेच", "मात्र", "त्यामुळे", "अनेक", "जास्त", "कमी", "करणार", "केल्या", "येथे", "तेथे",
}

# English Stopwords Fallback
ENGLISH_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is",
    "was", "are", "were", "been", "by", "from", "that", "this", "which", "as", "it",
    "be", "has", "have", "had", "will", "would", "can", "could", "should", "not",
}


def detect_language(text: str) -> str:
    """
    Detect language: 'hi' (Hindi), 'mr' (Marathi), or 'en' (English / other).
    """
    if not text:
        return "en"

    # Count Devanagari characters
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    total_chars = len(re.sub(r"\s+", "", text))

    if total_chars == 0 or (devanagari_chars / total_chars) < 0.2:
        return "en"

    # Distinct Marathi markers
    words = set(re.findall(r"\b\w+\b", text.lower()))
    marathi_markers = {"आणि", "आहेत", "त्यांच्या", "म्हणून", "केले", "झाले", "झाला", "करणार", "त्यामुळे", "झाली", "केली"}
    hindi_markers = {"हैं", "था", "थे", "थी", "किया", "गया", "सकते", "सकता", "सकती", "अपने", "उन्हें"}

    marathi_overlap = len(words.intersection(marathi_markers))
    hindi_overlap = len(words.intersection(hindi_markers))

    if marathi_overlap > hindi_overlap:
        return "mr"
    elif hindi_overlap > marathi_overlap:
        return "hi"
    
    # Check for Marathi 'ळ' (U+0933) which is highly characteristic of Marathi
    if "ळ" in text:
        return "mr"

    return "hi"


def get_language_display_name(lang_code: str) -> str:
    """Return friendly language label."""
    mapping = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "mr": "Marathi (मराठी)",
    }
    return mapping.get(lang_code, "English")


def get_stopwords_for_language(lang_code: str) -> List[str]:
    """Retrieve appropriate stopwords list."""
    if lang_code == "hi":
        return list(HINDI_STOPWORDS)
    elif lang_code == "mr":
        return list(MARATHI_STOPWORDS)
    return list(ENGLISH_STOPWORDS)
