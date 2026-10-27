"""
Unit tests for Multilingual NLP capabilities (English, Hindi, Marathi).
"""

import pytest
from src.summarization.multilingual import detect_language, get_stopwords_for_language
from src.summarization.preprocessing import split_into_sentences
from src.summarization.extractive import ExtractiveSummarizer


def test_detect_language():
    eng_text = "Modern artificial intelligence and transformers synthesize long documents."
    assert detect_language(eng_text) == "en"

    hindi_text = "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान मिशन के माध्यम से इतिहास रचा है। विक्रम लैंडर ने सफल लैंडिंग की।"
    assert detect_language(hindi_text) == "hi"

    marathi_text = "महाराष्ट्रातील शेतीमध्ये ठिबक सिंचन आणि आधुनिक तंत्रज्ञानाचा वापर मोठ्या प्रमाणावर केला जात आहे. त्यामुळे शेतकऱ्यांना फायदा होतो."
    assert detect_language(marathi_text) == "mr"


def test_devanagari_sentence_splitting():
    hindi_text = "पहला वाक्य यहाँ समाप्त होता है। दूसरा वाक्य आगे बढ़ता है। तीसरा वाक्य निष्कर्ष प्रस्तुत करता है।"
    sentences = split_into_sentences(hindi_text)
    assert len(sentences) == 3
    assert "पहला वाक्य" in sentences[0]
    assert "दूसरा वाक्य" in sentences[1]
    assert "तीसरा वाक्य" in sentences[2]


def test_extractive_summarization_marathi():
    marathi_doc = (
        "महाराष्ट्रातील ग्रामीण अर्थव्यवस्थेत कृषी क्षेत्राचे अनन्यसाधारण महत्त्व आहे. "
        "दुष्काळ आणि अनपेक्षित हवामान बदलांमुळे शेतकऱ्यांचे मोठे नुकसान होत असते. "
        "परंतु ठिबक सिंचन आणि सौर ऊर्जेवर चालणाऱ्या कृषी पंपांमुळे शेतीत लक्षणीय सुधारणा झाली आहे. "
        "राज्य शासनाच्या विविध योजनांचा लाभ घेऊन शेतकरी आता समृद्ध होत आहेत."
    )
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(marathi_doc, num_sentences=2)

    assert res["summary"] != ""
    assert len(res["selected_sentences"]) == 2
    assert res["language"] == "mr"
    assert res["total_sentences"] == 4


def test_extractive_summarization_hindi():
    hindi_doc = (
        "चंद्रयान मिशन भारत की वैज्ञानिक प्रगति का एक प्रमुख ऐतिहासिक प्रतीक है। "
        "विक्रम लैंडर ने चंद्रमा के दक्षिणी ध्रुव पर सफलतापूर्वक कदम रखा। "
        "प्रज्ञान रोवर ने चंद्रमा की मिट्टी में सल्फर और अन्य महत्वपूर्ण खनिजों की खोज की। "
        "इस उपलब्धि से संपूर्ण विश्व में भारत का वैज्ञानिक गौरव बढ़ा है।"
    )
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(hindi_doc, num_sentences=2)

    assert res["summary"] != ""
    assert len(res["selected_sentences"]) == 2
    assert res["language"] == "hi"
    assert res["total_sentences"] == 4
