"""
Data module containing curated sample articles and dataset loaders.
"""

from .sample_articles import get_sample_articles, get_sample_by_id, SAMPLE_ARTICLES
from .dataset_loader import load_cnn_dailymail_samples, FALLBACK_CNN_DAILYMAIL_SAMPLES

__all__ = [
    "get_sample_articles",
    "get_sample_by_id",
    "SAMPLE_ARTICLES",
    "load_cnn_dailymail_samples",
    "FALLBACK_CNN_DAILYMAIL_SAMPLES",
]
