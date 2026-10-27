"""
ROUGE Evaluation Engine for TextSummarize AI.
Calculates ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1) scaled 0-100.
Supports Google rouge-score library and pure Python verified algorithm fallback.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re
from collections import Counter


@dataclass
class MetricScore:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 2),
            "recall": round(self.recall, 2),
            "f1": round(self.f1, 2),
        }


@dataclass
class RougeScoresResult:
    has_reference: bool
    rouge1: MetricScore
    rouge2: MetricScore
    rougeL: MetricScore
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if not self.has_reference:
            return {
                "has_reference": False,
                "message": self.message or "Reference summary missing. ROUGE scores cannot be computed.",
            }
        return {
            "has_reference": True,
            "rouge1": self.rouge1.to_dict(),
            "rouge2": self.rouge2.to_dict(),
            "rougeL": self.rougeL.to_dict(),
        }


def _tokenize(text: str) -> List[str]:
    """Tokenize lowercase words removing punctuation."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract consecutive n-grams from token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(tokens1: List[str], tokens2: List[str]) -> int:
    """Compute length of Longest Common Subsequence using DP."""
    n, m = len(tokens1), len(tokens2)
    if n == 0 or m == 0:
        return 0
    # Space-optimized DP
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_pure_python_rouge(candidate: str, reference: str) -> RougeScoresResult:
    """Standard algorithmic implementation of ROUGE-1, ROUGE-2, and ROUGE-L."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        empty = MetricScore(0.0, 0.0, 0.0)
        return RougeScoresResult(True, empty, empty, empty)

    def calc_rouge_n(n: int) -> MetricScore:
        cand_ngrams = _get_ngrams(cand_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not cand_ngrams or not ref_ngrams:
            return MetricScore(0.0, 0.0, 0.0)

        cand_counter = Counter(cand_ngrams)
        ref_counter = Counter(ref_ngrams)

        overlap = sum(min(count, ref_counter[ng]) for ng, count in cand_counter.items())
        prec = (overlap / len(cand_ngrams)) * 100.0 if len(cand_ngrams) > 0 else 0.0
        rec = (overlap / len(ref_ngrams)) * 100.0 if len(ref_ngrams) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return MetricScore(prec, rec, f1)

    # ROUGE-1 & ROUGE-2
    r1 = calc_rouge_n(1)
    r2 = calc_rouge_n(2)

    # ROUGE-L
    lcs_len = _lcs_length(cand_tokens, ref_tokens)
    prec_l = (lcs_len / len(cand_tokens)) * 100.0 if len(cand_tokens) > 0 else 0.0
    rec_l = (lcs_len / len(ref_tokens)) * 100.0 if len(ref_tokens) > 0 else 0.0
    f1_l = (2 * prec_l * rec_l) / (prec_l + rec_l) if (prec_l + rec_l) > 0 else 0.0
    rL = MetricScore(prec_l, rec_l, f1_l)

    return RougeScoresResult(has_reference=True, rouge1=r1, rouge2=r2, rougeL=rL)


def compute_rouge_scores(candidate_summary: str, reference_summary: Optional[str]) -> RougeScoresResult:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L between candidate and reference summaries.
    Returns structured results scaled 0-100.
    """
    if not reference_summary or not reference_summary.strip():
        empty = MetricScore(0.0, 0.0, 0.0)
        return RougeScoresResult(
            has_reference=False,
            rouge1=empty,
            rouge2=empty,
            rougeL=empty,
            message="No ground-truth reference summary provided. ROUGE metrics require a genuine reference summary for comparison.",
        )

    if not candidate_summary or not candidate_summary.strip():
        empty = MetricScore(0.0, 0.0, 0.0)
        return RougeScoresResult(
            has_reference=True,
            rouge1=empty,
            rouge2=empty,
            rougeL=empty,
            message="Candidate summary is empty. Scores are 0.",
        )

    # Check if text contains Indic / Devanagari characters
    is_indic = bool(re.search(r"[\u0900-\u097F]", reference_summary + candidate_summary))
    if is_indic:
        return compute_pure_python_rouge(candidate_summary, reference_summary)

    # Attempt to use Google rouge_score library if available for English
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference_summary, candidate_summary)

        r1 = MetricScore(
            precision=scores["rouge1"].precision * 100.0,
            recall=scores["rouge1"].recall * 100.0,
            f1=scores["rouge1"].fmeasure * 100.0,
        )
        r2 = MetricScore(
            precision=scores["rouge2"].precision * 100.0,
            recall=scores["rouge2"].recall * 100.0,
            f1=scores["rouge2"].fmeasure * 100.0,
        )
        rL = MetricScore(
            precision=scores["rougeL"].precision * 100.0,
            recall=scores["rougeL"].recall * 100.0,
            f1=scores["rougeL"].fmeasure * 100.0,
        )
        return RougeScoresResult(has_reference=True, rouge1=r1, rouge2=r2, rougeL=rL)
    except Exception:
        # Transparent fallback to mathematical ROUGE algorithm
        return compute_pure_python_rouge(candidate_summary, reference_summary)
