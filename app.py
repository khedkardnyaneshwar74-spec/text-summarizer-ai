"""
TextSummarize AI - Intelligent Text Summarization & Evaluation Platform
Main Application & Interactive Overview Dashboard
"""

import streamlit as st
import torch
import sys
import os

from src.utils.model_specs import (
    get_model_limitations_dataframe,
    render_model_limitations_cards_html,
)

# Set page configuration
st.set_page_config(
    page_title="TextSummarize AI | Intelligent Summarization & Evaluation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, vibrant gradients, and glassmorphism
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero Banner with Multi-Gradient Glow */
    .hero-card {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.3) 0%, rgba(15, 23, 42, 0) 45%),
                    radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.25) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 20px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.4), 0 0 25px rgba(99, 102, 241, 0.15);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 40%, #F472B6 80%, #FB7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.18rem;
        color: #94A3B8;
        line-height: 1.65;
        margin-bottom: 1.4rem;
        max-width: 950px;
    }

    /* Feature Badges with Dynamic Accents */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.84rem;
        font-weight: 600;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* Method Highlight Cards */
    .method-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
        height: 100%;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .method-card:hover {
        border-color: #6366F1;
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
    }
    .method-card h3 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .method-desc {
        color: #94A3B8;
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* Stat Highlight Tags */
    .stat-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
    }

    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.6rem;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
        box-shadow: 0 6px 22px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize shared Session State
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "source_doc_input" not in st.session_state:
    st.session_state.source_doc_input = ""
if "reference_summary" not in st.session_state:
    st.session_state.reference_summary = ""
if "extractive_summary" not in st.session_state:
    st.session_state.extractive_summary = ""
if "abstractive_summary" not in st.session_state:
    st.session_state.abstractive_summary = ""
if "extractive_metadata" not in st.session_state:
    st.session_state.extractive_metadata = {}
if "abstractive_metadata" not in st.session_state:
    st.session_state.abstractive_metadata = {}
if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = "distilbart"

# Sidebar Navigation and System Status
with st.sidebar:
    st.markdown("### ⚙️ System Environment")
    device_status = "⚡ CUDA (GPU)" if torch.cuda.is_available() else "💻 CPU (Optimized)"
    st.info(f"**Hardware:** {device_status}\n\n**Python:** {sys.version.split()[0]}\n\n**PyTorch:** {torch.__version__}")
    
    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    st.markdown(
        """
        - [📝 **Summarize Document**](Summarize)
        - [📊 **Compare Methods & ROUGE**](Compare_Methods)
        - [🧪 **Dataset Benchmark**](Dataset_Benchmark)
        - [ℹ️ **Architecture & About**](About)
        """
    )
    st.markdown("---")
    st.markdown("### 🌐 Multilingual & Voice")
    st.caption("Supports English, Hindi (हिंदी), and Marathi (मराठी) with Real-Time Speech-to-Text & Audio Synthesis.")

# Hero Header Banner
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">TextSummarize AI</div>
        <div class="hero-subtitle">
            An intelligent Natural Language Processing platform synthesizing and evaluating 
            <strong>TF-IDF Extractive</strong> sentence salience ranking alongside modern 
            <strong>Transformer Seq2Seq Abstractive</strong> architectures (DistilBART, BART, T5, mT5) with 
            standardized <strong>ROUGE-1, ROUGE-2, and ROUGE-L</strong> scoring across English, Hindi, and Marathi documents.
        </div>
        <div>
            <span class="badge-pill">🚀 Pretrained Transformers (DistilBART, BART, T5)</span>
            <span class="badge-pill">📐 TF-IDF Salience Ranking</span>
            <span class="badge-pill">📊 Genuine ROUGE-1/2/L Evaluation</span>
            <span class="badge-pill">📰 CNN/DailyMail Benchmarking</span>
            <span class="badge-pill">🌐 Multilingual (EN, HI, MR)</span>
            <span class="badge-pill">🎙️ Voice Input & Whisper Audio Transcription</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Core Value Method Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="method-card">
            <h3>🔹 Extractive Summarization</h3>
            <div class="method-desc">
                <strong>Technique:</strong> Statistical TF-IDF Vectorization & Sentence Salience Ranking.<br><br>
                Identifies key informativeness signals across document sentences and reconstructs them in their 
                <strong>original chronological order</strong> to preserve natural reading narrative with zero risk of factual hallucination.
                <br><br>
                <span class="stat-pill">Latency: ~0.02s</span> &nbsp;
                <span class="stat-pill">100% Factually Faithful</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="method-card">
            <h3>🔸 Abstractive Summarization</h3>
            <div class="method-desc">
                <strong>Technique:</strong> Pretrained Seq2Seq Transformers (DistilBART / BART-CNN / T5).<br><br>
                Synthesizes novel phrasing and paraphrases core semantic arguments using 
                <strong>deterministic beam search</strong> to create concise, human-style executive summaries.
                <br><br>
                <span class="stat-pill">Neural Encoder-Decoder</span> &nbsp;
                <span class="stat-pill">Fluid Paraphrasing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Workflow Overview
st.markdown("### 🔄 Summarization & Evaluation Workflow")

step_col1, step_col2, step_col3, step_col4 = st.columns(4)

with step_col1:
    st.markdown(
        """
        **1. Input, Voice & Audio**  
        Type, paste, dictate live voice, or upload recorded `.wav` audio in English, Hindi, or Marathi.
        """
    )

with step_col2:
    st.markdown(
        """
        **2. Dual Processing**  
        Both extractive & abstractive models process the exact same source document in parallel.
        """
    )

with step_col3:
    st.markdown(
        """
        **3. ROUGE Scoring**  
        Genuine ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 calculation against reference summaries.
        """
    )

with step_col4:
    st.markdown(
        """
        **4. Visual Analytics & Export**  
        Interactive comparison charts, compression curves, audio playback, and 1-click TXT, JSON, or CSV downloads.
        """
    )

st.markdown("---")

# Quick CTA Navigation
cta_col1, cta_col2 = st.columns([1, 1])
with cta_col1:
    if st.button("🚀 Open Summarization Workspace", use_container_width=True):
        st.switch_page("pages/1_Summarize.py")

with cta_col2:
    if st.button("🧪 Run CNN/DailyMail Benchmark", use_container_width=True):
        st.switch_page("pages/3_Dataset_Benchmark.py")

# Model Architecture & Limitations Specifications Table
st.markdown("---")
st.markdown("### 📋 Model Architecture Specifications & Operational Limitations Catalog")
st.caption("Standardized architectural breakdown specifying token limits, hardware demands, and hallucination risks for every supported model.")

df_limits = get_model_limitations_dataframe()
st.dataframe(df_limits, use_container_width=True, hide_index=True)

with st.expander("🔍 Detailed Architectural Strengths & Risk Cards", expanded=False):
    st.markdown(render_model_limitations_cards_html(), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 1.2rem 0;">
        <strong>TextSummarize AI</strong> • Academic NLP Mini-Project • MIT License<br>
        Built with Streamlit, Hugging Face Transformers, PyTorch, scikit-learn, and Plotly.
    </div>
    """,
    unsafe_allow_html=True,
)
