"""
Benchmark runner for TextSummarize AI.
Executes dual summarization and ROUGE evaluation across dataset samples,
aggregating genuine performance metrics and exporting structured reports.
"""

from typing import List, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd

from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer
from src.evaluation.rouge_metrics import compute_rouge_scores
from src.utils.text_statistics import compute_compression_ratio


def run_benchmark_on_samples(
    samples: List[Dict[str, Any]],
    extractive_summarizer: Optional[ExtractiveSummarizer] = None,
    abstractive_summarizer: Optional[AbstractiveSummarizer] = None,
    num_extractive_sentences: int = 3,
    abstractive_max_length: int = 130,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run comparative benchmark across a collection of article/reference samples.

    Returns:
        Dict with:
            - records: List of per-sample detailed evaluation dicts
            - aggregate: Summary statistics (mean R1, R2, RL F1, latency, compression)
            - total_samples: Count of samples tested
            - successful_samples: Count of successfully evaluated samples
            - failed_samples: Count of failed samples
    """
    if extractive_summarizer is None:
        extractive_summarizer = ExtractiveSummarizer()

    records: List[Dict[str, Any]] = []
    total = len(samples)
    successful = 0
    failed = 0

    for i, item in enumerate(samples):
        sample_id = item.get("id", f"sample_{i+1}")
        article = item.get("article", "").strip()
        reference = item.get("highlights", "") or item.get("reference_summary", "").strip()

        if progress_callback:
            progress_callback(i + 1, total, f"Evaluating sample {i+1}/{total} ({sample_id})...")

        if not article or not reference:
            failed += 1
            continue

        try:
            # 1. Extractive Generation
            ext_res = extractive_summarizer.summarize(article, num_sentences=num_extractive_sentences)
            ext_summary = ext_res["summary"]
            ext_time = ext_res["execution_time"]

            # 2. Abstractive Generation
            abs_time = 0.0
            abs_summary = ""
            if abstractive_summarizer is not None:
                abs_res = abstractive_summarizer.summarize(article, max_length=abstractive_max_length)
                abs_summary = abs_res["summary"]
                abs_time = abs_res["execution_time"]

            # 3. ROUGE Evaluation
            ext_rouge = compute_rouge_scores(ext_summary, reference)
            abs_rouge = compute_rouge_scores(abs_summary, reference) if abs_summary else None

            # Word counts and compression
            orig_words = len(article.split())
            ext_words = len(ext_summary.split())
            abs_words = len(abs_summary.split()) if abs_summary else 0

            ext_comp = compute_compression_ratio(orig_words, ext_words)
            abs_comp = compute_compression_ratio(orig_words, abs_words) if abs_summary else 0.0

            record = {
                "sample_id": sample_id,
                "dataset": item.get("dataset", "Unknown"),
                "original_words": orig_words,
                "ref_words": len(reference.split()),
                # Extractive metrics
                "ext_words": ext_words,
                "ext_latency_sec": round(ext_time, 3),
                "ext_compression_pct": ext_comp,
                "ext_rouge1_f1": ext_rouge.rouge1.f1 if ext_rouge.has_reference else 0.0,
                "ext_rouge2_f1": ext_rouge.rouge2.f1 if ext_rouge.has_reference else 0.0,
                "ext_rougeL_f1": ext_rouge.rougeL.f1 if ext_rouge.has_reference else 0.0,
                # Abstractive metrics
                "abs_words": abs_words,
                "abs_latency_sec": round(abs_time, 3),
                "abs_compression_pct": abs_comp,
                "abs_rouge1_f1": abs_rouge.rouge1.f1 if (abs_rouge and abs_rouge.has_reference) else 0.0,
                "abs_rouge2_f1": abs_rouge.rouge2.f1 if (abs_rouge and abs_rouge.has_reference) else 0.0,
                "abs_rougeL_f1": abs_rouge.rougeL.f1 if (abs_rouge and abs_rouge.has_reference) else 0.0,
                # Texts for inspection
                "ext_summary": ext_summary,
                "abs_summary": abs_summary,
                "reference_summary": reference,
            }
            records.append(record)
            successful += 1
        except Exception as e:
            failed += 1
            records.append({
                "sample_id": sample_id,
                "error": str(e),
                "status": "FAILED",
            })

    # Compute aggregate means across successfully evaluated samples
    valid_records = [r for r in records if "error" not in r]
    aggregate = {}
    if valid_records:
        df = pd.DataFrame(valid_records)
        aggregate = {
            "evaluated_samples": len(valid_records),
            "ext_mean_rouge1_f1": round(float(df["ext_rouge1_f1"].mean()), 2),
            "ext_mean_rouge2_f1": round(float(df["ext_rouge2_f1"].mean()), 2),
            "ext_mean_rougeL_f1": round(float(df["ext_rougeL_f1"].mean()), 2),
            "ext_avg_latency_sec": round(float(df["ext_latency_sec"].mean()), 3),
            "ext_avg_compression_pct": round(float(df["ext_compression_pct"].mean()), 2),
            "abs_mean_rouge1_f1": round(float(df["abs_rouge1_f1"].mean()), 2),
            "abs_mean_rouge2_f1": round(float(df["abs_rouge2_f1"].mean()), 2),
            "abs_mean_rougeL_f1": round(float(df["abs_rougeL_f1"].mean()), 2),
            "abs_avg_latency_sec": round(float(df["abs_latency_sec"].mean()), 3),
            "abs_avg_compression_pct": round(float(df["abs_compression_pct"].mean()), 2),
        }

    return {
        "records": records,
        "aggregate": aggregate,
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": failed,
    }
