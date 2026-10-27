"""
Unit tests for ROUGE evaluation engine.
"""

import pytest
from src.evaluation.rouge_metrics import compute_rouge_scores


def test_rouge_with_identical_text():
    text = "The rapid development of renewable solar energy transforms the electric power grid."
    result = compute_rouge_scores(text, text)

    assert result.has_reference is True
    # Identical texts must yield 100% precision, recall, and F1
    assert result.rouge1.f1 == pytest.approx(100.0, rel=1e-2)
    assert result.rouge2.f1 == pytest.approx(100.0, rel=1e-2)
    assert result.rougeL.f1 == pytest.approx(100.0, rel=1e-2)


def test_rouge_with_missing_reference():
    candidate = "This summary has no reference to compare against."
    result = compute_rouge_scores(candidate, None)

    assert result.has_reference is False
    assert "No ground-truth reference summary provided" in result.message
    assert result.rouge1.f1 == 0.0

    result_empty_str = compute_rouge_scores(candidate, "   ")
    assert result_empty_str.has_reference is False


def test_rouge_with_empty_candidate():
    reference = "Ground truth reference summary."
    result = compute_rouge_scores("", reference)

    assert result.has_reference is True
    assert result.rouge1.f1 == 0.0
    assert result.rouge2.f1 == 0.0
    assert result.rougeL.f1 == 0.0


def test_rouge_partial_overlap():
    candidate = "Researchers develop innovative solar battery cells for clean energy."
    reference = "Scientists develop solar cells and battery storage systems for renewable clean energy."

    result = compute_rouge_scores(candidate, reference)
    assert result.has_reference is True
    assert 0.0 < result.rouge1.f1 <= 100.0
    assert 0.0 < result.rougeL.f1 <= 100.0


def test_rouge_multilingual_hindi_marathi():
    hindi_cand = "चंद्रयान मिशन ने चंद्रमा के दक्षिणी ध्रुव पर सफल लैंडिंग की।"
    hindi_ref = "इसरो के चंद्रयान मिशन ने चंद्रमा के ध्रुव पर ऐतिहासिक लैंडिंग की।"
    hindi_res = compute_rouge_scores(hindi_cand, hindi_ref)
    assert hindi_res.has_reference is True
    assert hindi_res.rouge1.f1 > 50.0

    marathi_cand = "महाराष्ट्रातील शेतीत आधुनिक कृषी तंत्रज्ञानाचा वापर वाढत आहे।"
    marathi_ref = "महाराष्ट्रात कृषी तंत्रज्ञानाचा वापर आणि जलसंधारण यशस्वी झाले आहे।"
    marathi_res = compute_rouge_scores(marathi_cand, marathi_ref)
    assert marathi_res.has_reference is True
    assert marathi_res.rouge1.f1 > 40.0
