"""
Export utilities for TextSummarize AI.
Provides clean TXT, JSON, and CSV export functionality.
"""

import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime


def export_summary_to_txt(
    original_text: str,
    extractive_summary: str,
    abstractive_summary: str,
    model_name: str = "BART",
    metrics: Dict[str, Any] = None,
) -> str:
    """Format full summarization comparison into an easily readable plain-text report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "TEXTSUMMARIZE AI - INTELLIGENT SUMMARY REPORT",
        f"Generated at: {timestamp}",
        "=" * 70,
        "",
        "--- ABSTRACTIVE SUMMARY (Transformer: " + model_name + ") ---",
        abstractive_summary.strip() if abstractive_summary else "[Not generated]",
        "",
        "--- EXTRACTIVE SUMMARY (TF-IDF Sentence Ranking) ---",
        extractive_summary.strip() if extractive_summary else "[Not generated]",
        "",
    ]

    if metrics:
        lines.append("--- EVALUATION METRICS (ROUGE) ---")
        for method, score_dict in metrics.items():
            lines.append(f"[{method}]")
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    lines.append(f"  {k}: {v}")
            lines.append("")

    lines.extend([
        "-" * 70,
        "ORIGINAL SOURCE DOCUMENT EXCERPT (First 500 chars):",
        (original_text[:500] + "...") if len(original_text) > 500 else original_text,
        "=" * 70,
    ])
    return "\n".join(lines)


def export_summary_to_json(
    original_text: str,
    extractive_summary: str,
    abstractive_summary: str,
    model_name: str,
    reference_summary: str = "",
    metrics: Dict[str, Any] = None,
    timings: Dict[str, float] = None,
) -> str:
    """Format full results into standard JSON."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "original_text_length_words": len(original_text.split()),
        "original_text_length_chars": len(original_text),
        "reference_summary": reference_summary or None,
        "extractive_summary": {
            "text": extractive_summary,
            "word_count": len(extractive_summary.split()) if extractive_summary else 0,
            "latency_seconds": timings.get("extractive", 0.0) if timings else 0.0,
        },
        "abstractive_summary": {
            "text": abstractive_summary,
            "word_count": len(abstractive_summary.split()) if abstractive_summary else 0,
            "latency_seconds": timings.get("abstractive", 0.0) if timings else 0.0,
        },
        "metrics": metrics or {},
    }
    return json.dumps(payload, indent=2)


def export_benchmark_to_csv(benchmark_records: List[Dict[str, Any]]) -> str:
    """Convert benchmark sample results list to CSV string."""
    df = pd.DataFrame(benchmark_records)
    return df.to_csv(index=False)
