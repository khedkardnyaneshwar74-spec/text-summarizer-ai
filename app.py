"""
TextSummarize AI - Intelligent Text Summarization & Evaluation Platform
Main Application & Interactive Overview Dashboard
"""

import streamlit as st
import torch
import sys
import os

# Set page configuration
st.set_page_config(
    page_title="TextSummarize AI | Intelligent Summarization & Evaluation",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics
st.markdown(
    """
    <style>
    /* Global Fonts & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    /* Feature Badge */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Method Cards */
    .method-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.6rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .method-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .method-card h3 {
        color: #F8FAFC;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .method-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* Stat Highlight */
    .stat-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize shared Session State
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
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

# Hero Header
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">TextSummarize AI</div>
        <div class="hero-subtitle">
            An intelligent NLP platform comparing traditional <strong>TF-IDF Extractive</strong> sentence ranking 
            with modern <strong>Transformer Abstractive</strong> generation (BART & T5) through standardized 
            <strong>ROUGE-1, ROUGE-2, and ROUGE-L</strong> metrics across English, Hindi, and Marathi documents.
        </div>
        <div>
            <span class="badge-pill">🚀 Pretrained Transformers</span>
            <span class="badge-pill">📐 TF-IDF Ranking</span>
            <span class="badge-pill">📊 Genuine ROUGE Evaluation</span>
            <span class="badge-pill">📰 CNN/DailyMail Benchmarking</span>
            <span class="badge-pill">🌐 Multilingual (EN, HI, MR)</span>
            <span class="badge-pill">🎙️ Voice Input & Audio Playback</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Core Value Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="method-card">
            <h3>🔹 Extractive Summarization</h3>
            <div class="method-desc">
                <strong>Technique:</strong> Statistical TF-IDF Vectorization & Sentence Salience Ranking.<br><br>
                Identifies key informativeness signals across document sentences and reconstructs them in their 
                <strong>original chronological order</strong> to preserve natural reading narrative.
                <br><br>
                <span class="stat-pill">Latency: ~0.01s</span> &nbsp;
                <span class="stat-pill">Interpretable & Verifiable</span>
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
                <strong>Technique:</strong> Pretrained Seq2Seq Transformer (BART-CNN / DistilBART / T5).<br><br>
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

# Workflow overview
st.markdown("### 🔄 Summarization & Evaluation Workflow")

step_col1, step_col2, step_col3, step_col4 = st.columns(4)

with step_col1:
    st.markdown(
        """
        **1. Input & Cleaning**  
        Paste text or upload `.txt` files. System validates bounds and normalizes text structure.
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
        Interactive comparison charts, compression rates, and 1-click TXT, JSON, or CSV downloads.
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;">
        <strong>TextSummarize AI</strong> • Academic NLP Mini-Project • MIT License<br>
        Built with Streamlit, Hugging Face Transformers, PyTorch, scikit-learn, and Plotly.
    </div>
    """,
    unsafe_allow_html=True,
)
