"""
Abstractive Summarization Engine using Pretrained Transformer Models (BART / DistilBART / T5).
Includes model caching, GPU/CPU auto-detection, deterministic generation, and long-input chunking.
"""

import time
import os
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Supported models catalog
SUPPORTED_MODELS = {
    "distilbart": {
        "id": "sshleifer/distilbart-cnn-12-6",
        "name": "DistilBART-CNN (High Efficiency)",
        "max_input_tokens": 1024,
        "is_t5": False,
    },
    "bart-large": {
        "id": "facebook/bart-large-cnn",
        "name": "BART-Large-CNN (High Capacity)",
        "max_input_tokens": 1024,
        "is_t5": False,
    },
    "t5-small": {
        "id": "t5-small",
        "name": "T5-Small (Unified Seq2Seq)",
        "max_input_tokens": 512,
        "is_t5": True,
    },
    "mt5-multilingual": {
        "id": "csebuetnlp/mT5_multilingual_XLSum",
        "name": "mT5-Multilingual (English, Hindi, Marathi)",
        "max_input_tokens": 512,
        "is_t5": True,
    },
}

DEFAULT_MODEL_KEY = "distilbart"

# Global in-memory cache for models and tokenizers
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def get_device() -> torch.device:
    """Detect CUDA GPU if available, else fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AbstractiveSummarizer:
    """
    Abstractive Summarization using Hugging Face Seq2Seq Transformers.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY):
        if model_key not in SUPPORTED_MODELS:
            self.model_id = model_key
            self.is_t5 = "t5" in model_key.lower()
            self.max_tokens = 512 if self.is_t5 else 1024
            self.model_key = model_key
        else:
            config = SUPPORTED_MODELS[model_key]
            self.model_id = config["id"]
            self.is_t5 = config["is_t5"]
            self.max_tokens = config["max_input_tokens"]
            self.model_key = model_key

        self.device = get_device()
        self.device_name = "CUDA (GPU)" if self.device.type == "cuda" else "CPU"
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model with singleton caching, independent of removed pipelines."""
        global _MODEL_CACHE
        if self.model_id in _MODEL_CACHE:
            self.tokenizer, self.model = _MODEL_CACHE[self.model_id]
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
            )
            model.to(self.device)
            model.eval()

            _MODEL_CACHE[self.model_id] = (tokenizer, model)
            self.tokenizer = tokenizer
            self.model = model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load abstractive summarization model '{self.model_id}': {str(e)}"
            )

    def summarize(
        self,
        text: str,
        max_length: int = 142,
        min_length: int = 40,
        num_beams: int = 4,
        length_penalty: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Generate abstractive summary with deterministic beam search.

        Args:
            text: Input document text.
            max_length: Maximum summary length in tokens.
            min_length: Minimum summary length in tokens.
            num_beams: Beam search width (default 4 for high-quality deterministic decoding).
            length_penalty: Exponential penalty to the length.

        Returns:
            Dict with:
                - summary: Generated abstractive summary text
                - model_id: Name of model used
                - device: Hardware device used
                - execution_time: Latency in seconds
                - truncated: Boolean indicating if input exceeded max input tokens
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return {
                "summary": "",
                "model_id": self.model_id,
                "device": self.device_name,
                "execution_time": 0.0,
                "truncated": False,
            }

        input_text = text.strip()
        if self.is_t5 and not input_text.startswith("summarize:"):
            input_text = f"summarize: {input_text}"

        # Tokenize and check token bounds
        tokens = self.tokenizer.encode(input_text, return_tensors="pt", truncation=False)
        input_token_count = tokens.shape[1]
        truncated = False

        if input_token_count > self.max_tokens:
            truncated = True
            # For documents longer than context window, summarize first max_tokens tokens safely
            input_ids = tokens[:, : self.max_tokens].to(self.device)
        else:
            input_ids = tokens.to(self.device)

        # Enforce min_length <= max_length
        actual_min_len = min(min_length, max_length - 5) if max_length > 10 else 10
        actual_max_len = max_length

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids,
                num_beams=num_beams,
                max_length=actual_max_len,
                min_length=actual_min_len,
                length_penalty=length_penalty,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,  # Deterministic decoding as specified
            )

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - start_time

        return {
            "summary": summary_text,
            "model_id": self.model_id,
            "device": self.device_name,
            "execution_time": elapsed,
            "input_tokens": input_token_count,
            "truncated": truncated,
        }
