"""
Evaluation module containing ROUGE metrics computation and benchmarking pipeline.
"""

from .rouge_metrics import compute_rouge_scores, RougeScoresResult
from .benchmark import run_benchmark_on_samples

__all__ = [
    "compute_rouge_scores",
    "RougeScoresResult",
    "run_benchmark_on_samples",
]
