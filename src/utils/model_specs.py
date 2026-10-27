"""
Model Specifications & Operational Limitations Catalog.
Provides empirical data on architecture, context bounds, hardware footprints, and hallucination risks.
"""

import pandas as pd
from typing import List, Dict, Any

MODEL_LIMITATIONS_DATA = [
    {
        "Model / Option": "TF-IDF Extractive Ranking",
        "Paradigm": "Unsupervised Statistical Salience",
        "Supported Languages": "English, Hindi (हिंदी), Marathi (मराठी)",
        "Input Context Limit": "Unlimited (~100,000+ words, O(N))",
        "Hardware & Latency": "CPU (< 50MB RAM, ~0.02s)",
        "Core Strengths": "100% Factually faithful (Zero Hallucination). Preserves exact statistics, quotes, and names in chronological order.",
        "Operational Limitations & Risks": "Cannot paraphrase or generate novel sentences. Vulnerable to sentence fragmentation. Lacks semantic coreference understanding.",
        "Best Suited For": "Legal briefs, financial balance sheets, medical records, and audit reports where factual alteration is prohibited.",
    },
    {
        "Model / Option": "DistilBART-CNN-12-6",
        "Paradigm": "Distilled Neural Seq2Seq Transformer",
        "Supported Languages": "English (Primary)",
        "Input Context Limit": "1,024 tokens (~750 words)",
        "Hardware & Latency": "CPU / GPU (~1.2GB RAM, ~1.5s)",
        "Core Strengths": "High compression ratio, fluent human-like prose, captures central narrative themes. 2x faster than full BART.",
        "Operational Limitations & Risks": "Potential for neural hallucination (generating plausible but unverified facts). Inputs > 1,024 tokens are truncated. English-only.",
        "Best Suited For": "News articles, executive briefings, technology blog summaries, and everyday general-purpose distillation.",
    },
    {
        "Model / Option": "BART-Large-CNN",
        "Paradigm": "Deep Seq2Seq Transformer (406M params)",
        "Supported Languages": "English (Primary)",
        "Input Context Limit": "1,024 tokens (~750 words)",
        "Hardware & Latency": "CPU / GPU (~3.5GB RAM, ~4.5s)",
        "Core Strengths": "High linguistic sophistication, rich vocabulary, highest baseline ROUGE-1/2/L scores on CNN/DailyMail benchmarks.",
        "Operational Limitations & Risks": "Heavier compute footprint. Non-zero hallucination risk. Inputs strictly bounded at 1,024 tokens. English-only.",
        "Best Suited For": "Long-form journalism, academic abstracts, and editorial essays requiring sophisticated language composition.",
    },
    {
        "Model / Option": "T5-Small",
        "Paradigm": "Unified Text-to-Text Seq2Seq (60M params)",
        "Supported Languages": "English (Primary)",
        "Input Context Limit": "512 tokens (~380 words)",
        "Hardware & Latency": "CPU (< 500MB RAM, ~1.0s)",
        "Core Strengths": "Ultra-lightweight footprint, minimal memory requirements, fast inference across low-tier hardware.",
        "Operational Limitations & Risks": "Narrow 512 token context window. Summaries can be slightly generic or terse compared to BART. English-only.",
        "Best Suited For": "Resource-constrained micro-instances, rapid paragraph abstracts, and lightweight automated processing pipelines.",
    },
    {
        "Model / Option": "mT5-Multilingual-XLSum",
        "Paradigm": "Cross-Lingual Seq2Seq Transformer (300M params)",
        "Supported Languages": "English, Hindi (हिंदी), Marathi (मराठी) + 40 langs",
        "Input Context Limit": "512 tokens (~380 words)",
        "Hardware & Latency": "CPU / GPU (~2.5GB RAM, ~6.0s)",
        "Core Strengths": "Capable of true abstractive neural rewriting across Hindi and Marathi Devanagari scripts natively.",
        "Operational Limitations & Risks": "512 token context window. Longer generation latency on CPU. Requires strict beam penalty to prevent repetition in Indic text.",
        "Best Suited For": "Multilingual Indian regional news, regional governance documents, and multilingual cross-language summarization.",
    },
]


def get_model_limitations_dataframe() -> pd.DataFrame:
    """Return model limitations catalog as a structured Pandas DataFrame."""
    return pd.DataFrame(MODEL_LIMITATIONS_DATA)


def render_model_limitations_cards_html() -> str:
    """
    Renders an aesthetic, responsive CSS grid of model cards highlighting strengths and limitations.
    Uses modern glassmorphism styling with distinct color accents.
    """
    cards_html = []
    
    color_map = {
        "TF-IDF Extractive Ranking": {"badge": "#3B82F6", "bg": "rgba(59, 130, 246, 0.08)", "border": "rgba(59, 130, 246, 0.3)"},
        "DistilBART-CNN-12-6": {"badge": "#8B5CF6", "bg": "rgba(139, 92, 246, 0.08)", "border": "rgba(139, 92, 246, 0.3)"},
        "BART-Large-CNN": {"badge": "#6366F1", "bg": "rgba(99, 102, 241, 0.08)", "border": "rgba(99, 102, 241, 0.3)"},
        "T5-Small": {"badge": "#10B981", "bg": "rgba(16, 185, 129, 0.08)", "border": "rgba(16, 185, 129, 0.3)"},
        "mT5-Multilingual-XLSum": {"badge": "#EC4899", "bg": "rgba(236, 72, 153, 0.08)", "border": "rgba(236, 72, 153, 0.3)"},
    }

    for item in MODEL_LIMITATIONS_DATA:
        style = color_map.get(item["Model / Option"], {"badge": "#38BDF8", "bg": "rgba(56, 189, 248, 0.08)", "border": "rgba(56, 189, 248, 0.3)"})
        card = f"""
        <div style="
            background: {style['bg']};
            border: 1px solid {style['border']};
            border-radius: 14px;
            padding: 1.3rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {style['badge']}; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">
                        {item['Paradigm'].split()[0]}
                    </span>
                    <strong style="color: #F8FAFC; font-size: 1.1rem;">{item['Model / Option']}</strong>
                </div>
                <span style="color: #94A3B8; font-size: 0.82rem; font-family: monospace;">⚡ {item['Hardware & Latency']}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 0.8rem;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #94A3B8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Context Bound:</span><br>
                    <span style="color: #E2E8F0; font-size: 0.88rem; font-weight: 500;">{item['Input Context Limit']}</span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <span style="color: #94A3B8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Languages:</span><br>
                    <span style="color: #E2E8F0; font-size: 0.88rem; font-weight: 500;">{item['Supported Languages']}</span>
                </div>
            </div>

            <div style="margin-bottom: 0.6rem;">
                <span style="color: #34D399; font-weight: 600; font-size: 0.84rem;">✅ Core Strengths:</span>
                <span style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.5;"> {item['Core Strengths']}</span>
            </div>
            
            <div style="margin-bottom: 0.6rem;">
                <span style="color: #F87171; font-weight: 600; font-size: 0.84rem;">⚠️ Operational Limitations & Risks:</span>
                <span style="color: #CBD5E1; font-size: 0.86rem; line-height: 1.5;"> {item['Operational Limitations & Risks']}</span>
            </div>

            <div>
                <span style="color: #60A5FA; font-weight: 600; font-size: 0.84rem;">🎯 Best Suited For:</span>
                <span style="color: #E2E8F0; font-size: 0.86rem; font-style: italic;"> {item['Best Suited For']}</span>
            </div>
        </div>
        """
        cards_html.append(card)

    return "\n".join(cards_html)
