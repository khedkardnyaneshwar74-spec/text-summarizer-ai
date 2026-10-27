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
