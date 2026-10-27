"""
End-to-end integration tests for TextSummarize AI.
Verifies full data flow across validation, summarization, ROUGE evaluation, and export.
"""

import pytest
import json
from src.utils.validation import validate_input_text
from src.utils.text_statistics import compute_text_statistics, compute_compression_ratio
from src.utils.export import export_summary_to_txt, export_summary_to_json, export_benchmark_to_csv
from src.summarization.extractive import ExtractiveSummarizer
from src.evaluation.rouge_metrics import compute_rouge_scores
from src.evaluation.benchmark import run_benchmark_on_samples
from src.data.sample_articles import get_sample_articles


def test_full_extractive_pipeline():
    samples = get_sample_articles()
    sample = samples[0]

    # 1. Validation
    val = validate_input_text(sample["article"])
    assert val.is_valid
    assert val.word_count > 50

    # 2. Extractive Summarization
    ext_engine = ExtractiveSummarizer()
    res = ext_engine.summarize(val.cleaned_text, num_sentences=2)
    assert res["summary"] != ""
    assert len(res["selected_sentences"]) == 2
    assert res["execution_time"] >= 0.0

    # 3. ROUGE Scoring
    rouge = compute_rouge_scores(res["summary"], sample["reference_summary"])
    assert rouge.has_reference is True
    assert 0.0 < rouge.rouge1.f1 <= 100.0
    assert 0.0 <= rouge.rouge2.f1 <= 100.0
    assert 0.0 < rouge.rougeL.f1 <= 100.0

    # 4. Text stats and compression
    stats = compute_text_statistics(res["summary"], label="Extractive")
    assert stats["word_count"] < val.word_count
    comp = compute_compression_ratio(val.word_count, stats["word_count"])
    assert 0.0 < comp < 100.0

    # 5. Export TXT and JSON
    txt_export = export_summary_to_txt(
        sample["article"],
        res["summary"],
        "",
        model_name="TF-IDF",
        metrics={"Extractive": rouge.to_dict()},
    )
    assert "TEXTSUMMARIZE AI" in txt_export
    assert res["summary"] in txt_export

    json_export = export_summary_to_json(
        sample["article"],
        res["summary"],
        "",
        model_name="TF-IDF",
        reference_summary=sample["reference_summary"],
        metrics={"Extractive": rouge.to_dict()},
    )
    parsed = json.loads(json_export)
    assert parsed["original_text_length_words"] == val.word_count
    assert parsed["extractive_summary"]["text"] == res["summary"]


def test_benchmark_runner_pipeline():
    samples = get_sample_articles()[:2]
    ext_engine = ExtractiveSummarizer()

    # Run benchmark without abstractive model for fast unit testing
    results = run_benchmark_on_samples(
        samples=samples,
        extractive_summarizer=ext_engine,
        abstractive_summarizer=None,
        num_extractive_sentences=2,
    )

    assert results["total_samples"] == 2
    assert results["successful_samples"] == 2
    assert results["failed_samples"] == 0
    assert len(results["records"]) == 2

    agg = results["aggregate"]
    assert agg["evaluated_samples"] == 2
    assert agg["ext_mean_rouge1_f1"] > 0.0

    # Verify CSV export
    csv_str = export_benchmark_to_csv(results["records"])
    assert "sample_id" in csv_str
    assert "ext_rouge1_f1" in csv_str
