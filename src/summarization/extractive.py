"""
Extractive Summarization Engine using TF-IDF Sentence Ranking.
Ranks document sentences by information salience and restores original chronological order.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import split_into_sentences


class ExtractiveSummarizer:
    """
    TF-IDF based Extractive Summarizer.
    Calculates sentence salience scores and outputs key sentences in original document order.
    """

    def __init__(self, stop_words: str = "english"):
        self.stop_words = stop_words

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate an extractive summary for the provided text.

        Args:
            text: Input document string.
            num_sentences: Number of top sentences to select.
            ratio: Optional target ratio of sentences (e.g., 0.25 for top 25%).

        Returns:
            Dict containing:
                - summary: The joined summary string
                - selected_sentences: List of chosen sentences
                - total_sentences: Total sentences in original text
                - execution_time: Latency in seconds
                - sentence_scores: List of (index, sentence, score) tuples
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return {
                "summary": "",
                "selected_sentences": [],
                "total_sentences": 0,
                "execution_time": 0.0,
                "sentence_scores": [],
            }

        sentences = split_into_sentences(text)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return {
                "summary": "",
                "selected_sentences": [],
                "total_sentences": 0,
                "execution_time": time.perf_counter() - start_time,
                "sentence_scores": [],
            }

        # If document has fewer sentences than requested, return all sentences in order
        if ratio is not None:
            target_count = max(1, int(round(total_sentences * ratio)))
        else:
            target_count = max(1, min(num_sentences, total_sentences))

        if total_sentences <= target_count:
            elapsed = time.perf_counter() - start_time
            return {
                "summary": " ".join(sentences),
                "selected_sentences": sentences,
                "total_sentences": total_sentences,
                "execution_time": elapsed,
                "sentence_scores": [(i, s, 1.0) for i, s in enumerate(sentences)],
            }

        # Compute TF-IDF matrix over sentences
        try:
            vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                token_pattern=r"(?u)\b\w+\b",
                lowercase=True,
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Fallback if vocabulary is entirely stop words or empty
            elapsed = time.perf_counter() - start_time
            selected = sentences[:target_count]
            return {
                "summary": " ".join(selected),
                "selected_sentences": selected,
                "total_sentences": total_sentences,
                "execution_time": elapsed,
                "sentence_scores": [(i, s, 0.0) for i, s in enumerate(selected)],
            }

        # Calculate salience score per sentence
        # Score = sum of TF-IDF values divided by sentence length normalization to avoid penalizing concise sentences
        sentence_scores = []
        for i in range(total_sentences):
            row = tfidf_matrix.getrow(i).toarray().flatten()
            non_zero_vals = row[row > 0]
            if len(non_zero_vals) > 0:
                # Average weight of salient terms in the sentence
                score = float(np.mean(non_zero_vals) * np.log1p(len(non_zero_vals)))
            else:
                score = 0.0
            sentence_scores.append((i, sentences[i], score))

        # Rank sentences by score descending
        ranked_indices = sorted(
            range(total_sentences),
            key=lambda idx: sentence_scores[idx][2],
            reverse=True,
        )

        # Select top target_count sentence indices
        chosen_indices = ranked_indices[:target_count]

        # RESTORE ORIGINAL CHRONOLOGICAL ORDER to maintain coherent flow
        chosen_indices.sort()

        selected_sentences = [sentences[idx] for idx in chosen_indices]
        summary_text = " ".join(selected_sentences)
        elapsed = time.perf_counter() - start_time

        return {
            "summary": summary_text,
            "selected_sentences": selected_sentences,
            "total_sentences": total_sentences,
            "execution_time": elapsed,
            "sentence_scores": sentence_scores,
        }
