"""
Page 2: Compare Methods & ROUGE Evaluation Dashboard
Side-by-side comparative analysis of Extractive (TF-IDF) vs Abstractive (Transformer) summarization.
Includes Empirical ROUGE Metrics, Efficiency Benchmarks, and Model Limitations Catalog.
"""

import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.evaluation.rouge_metrics import compute_rouge_scores
from src.utils.text_statistics import compute_compression_ratio
from src.utils.model_specs import (
    get_model_limitations_dataframe,
    render_model_limitations_cards_html,
)

st.set_page_config(
    page_title="Compare Methods & ROUGE | TextSummarize AI",
    page_icon="📊",
    layout="wide",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .compare-banner {
        background: radial-gradient(circle at 15% 30%, rgba(59, 130, 246, 0.25) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 18px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .compare-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .compare-desc {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .output-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Retrieve data from session state
doc_text = st.session_state.get("document_text", "")
ext_summary = st.session_state.get("extractive_summary", "")
abs_summary = st.session_state.get("abstractive_summary", "")
ref_summary = st.session_state.get("reference_summary", "")
ext_meta = st.session_state.get("extractive_metadata", {})
abs_meta = st.session_state.get("abstractive_metadata", {})

# Header Banner
st.markdown(
    """
    <div class="compare-banner">
        <div class="compare-title">Comparative Analysis & Evaluation Dashboard</div>
        <div class="compare-desc">
            Empirical side-by-side evaluation contrasting <strong>Extractive TF-IDF Salience Ranking</strong> 
            against <strong>Neural Seq2Seq Transformers</strong> using standardized 
            <strong>ROUGE-1, ROUGE-2, and ROUGE-L</strong> metrics, compression curves, and latency profiles.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ext_summary and not abs_summary:
    st.info("💡 No summaries generated yet. Please visit the **Summarize** page first to input text or voice and generate summaries.")
    if st.button("👉 Open Summarization Workspace"):
        st.switch_page("pages/1_Summarize.py")
    st.stop()

# 1. Summary Overview Cards
st.markdown("### 🔍 Generated Outputs Inspection")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🔹 Extractive Output (TF-IDF)")
    if ext_summary:
        st.info(ext_summary)
    else:
        st.warning("[Extractive summary not generated]")
with c2:
    st.markdown("#### 🔸 Abstractive Output (Transformer)")
    if abs_summary:
        st.success(abs_summary)
    else:
        st.warning("[Abstractive summary not generated]")

# 2. Reference Summary & ROUGE Scores
st.markdown("---")
st.markdown("### 🎯 ROUGE Quality Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)")

if not ref_summary.strip():
    st.warning(
        "⚠️ **Reference-based ROUGE evaluation is currently awaiting a reference summary.**\n\n"
        "ROUGE requires a genuine human-written gold-standard reference summary to measure unigram, bigram, and longest common subsequence overlap. "
        "You can provide a reference summary below to compute real ROUGE scores right now."
    )
    new_ref = st.text_area("Enter Reference Summary to evaluate right now:", value="", height=100)
    if st.button("Evaluate ROUGE Now"):
        st.session_state.reference_summary = new_ref
        st.rerun()
else:
    st.markdown(f"**Ground-Truth Reference:** *{ref_summary}*")

    # Compute genuine ROUGE scores
    rouge_ext = compute_rouge_scores(ext_summary, ref_summary) if ext_summary else None
    rouge_abs = compute_rouge_scores(abs_summary, ref_summary) if abs_summary else None

    # Tabular presentation of Precision, Recall, F1
    table_rows = []
    if rouge_ext and rouge_ext.has_reference:
        table_rows.extend([
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-1 (Unigrams)", "Precision": rouge_ext.rouge1.precision, "Recall": rouge_ext.rouge1.recall, "F1-Score": rouge_ext.rouge1.f1},
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-2 (Bigrams)", "Precision": rouge_ext.rouge2.precision, "Recall": rouge_ext.rouge2.recall, "F1-Score": rouge_ext.rouge2.f1},
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-L (LCS)", "Precision": rouge_ext.rougeL.precision, "Recall": rouge_ext.rougeL.recall, "F1-Score": rouge_ext.rougeL.f1},
        ])
    if rouge_abs and rouge_abs.has_reference:
        table_rows.extend([
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-1 (Unigrams)", "Precision": rouge_abs.rouge1.precision, "Recall": rouge_abs.rouge1.recall, "F1-Score": rouge_abs.rouge1.f1},
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-2 (Bigrams)", "Precision": rouge_abs.rouge2.precision, "Recall": rouge_abs.rouge2.recall, "F1-Score": rouge_abs.rouge2.f1},
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-L (LCS)", "Precision": rouge_abs.rougeL.precision, "Recall": rouge_abs.rougeL.recall, "F1-Score": rouge_abs.rougeL.f1},
        ])

    if table_rows:
        df_rouge = pd.DataFrame(table_rows)
        st.dataframe(
            df_rouge.style.format({"Precision": "{:.2f}%", "Recall": "{:.2f}%", "F1-Score": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

        # ROUGE F1 Comparison Chart
        st.markdown("#### 📊 ROUGE F1 Comparison Chart")
        if HAS_PLOTLY:
            fig_f1 = px.bar(
                df_rouge,
                x="Metric",
                y="F1-Score",
                color="Method",
                barmode="group",
                text="F1-Score",
                color_discrete_map={"Extractive (TF-IDF)": "#38BDF8", "Abstractive (Transformer)": "#A78BFA"},
                title="ROUGE F1-Score Comparison (%)",
            )
            fig_f1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_f1.update_layout(yaxis_range=[0, 110], template="plotly_dark", height=400)
            st.plotly_chart(fig_f1, use_container_width=True)
        else:
            pivot_f1 = df_rouge.pivot(index="Metric", columns="Method", values="F1-Score")
            st.bar_chart(pivot_f1, height=350)

        st.caption(
            "ℹ️ **ROUGE Interpretation:** ROUGE scores quantify lexical n-gram and longest common subsequence overlap against the reference. "
            "Higher scores denote stronger token alignment with the gold standard."
        )

# 3. Document Statistics & Efficiency Metrics
st.markdown("---")
st.markdown("### 📈 Efficiency & Compression Analytics")

orig_words = len(doc_text.split()) if doc_text else 1
ext_words = len(ext_summary.split()) if ext_summary else 0
abs_words = len(abs_summary.split()) if abs_summary else 0

ext_comp = compute_compression_ratio(orig_words, ext_words)
abs_comp = compute_compression_ratio(orig_words, abs_words)

ext_lat = ext_meta.get("execution_time", 0.0)
abs_lat = abs_meta.get("execution_time", 0.0)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    stats_df = pd.DataFrame([
        {"Method": "Original Document", "Words": orig_words, "Compression (%)": 0.0},
        {"Method": "Extractive (TF-IDF)", "Words": ext_words, "Compression (%)": ext_comp},
        {"Method": "Abstractive (Transformer)", "Words": abs_words, "Compression (%)": abs_comp},
    ])
    if HAS_PLOTLY:
        fig_len = px.bar(
            stats_df,
            x="Method",
            y="Words",
            color="Method",
            text="Words",
            title="Summary Word Count vs Source Document",
            color_discrete_sequence=["#64748B", "#38BDF8", "#A78BFA"],
        )
        fig_len.update_traces(textposition="outside")
        fig_len.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_len, use_container_width=True)
    else:
        st.markdown("##### Word Count Comparison")
        st.bar_chart(stats_df.set_index("Method")["Words"], height=300)

with col_chart2:
    latency_df = pd.DataFrame([
        {"Method": "Extractive (TF-IDF)", "Latency (s)": ext_lat},
        {"Method": "Abstractive (Transformer)", "Latency (s)": abs_lat},
    ])
    if HAS_PLOTLY:
        fig_lat = px.bar(
            latency_df,
            x="Method",
            y="Latency (s)",
            color="Method",
            text="Latency (s)",
            title="Processing Latency (Seconds)",
            color_discrete_map={"Extractive (TF-IDF)": "#38BDF8", "Abstractive (Transformer)": "#A78BFA"},
        )
        fig_lat.update_traces(texttemplate="%{text:.3f}s", textposition="outside")
        fig_lat.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.markdown("##### Latency (Seconds)")
        st.bar_chart(latency_df.set_index("Method")["Latency (s)"], height=300)

# 4. Model Architecture & Limitations Catalog (Table + Cards)
st.markdown("---")
st.markdown("### 📋 Model Limitations & Operational Constraints Catalog")
st.caption("Standardized architectural breakdown specifying token limits, hardware demands, and hallucination risks for every supported model.")

df_limits = get_model_limitations_dataframe()
st.dataframe(df_limits, use_container_width=True, hide_index=True)

with st.expander("🔍 Detailed Architectural Strengths & Risk Cards", expanded=False):
    st.markdown(render_model_limitations_cards_html(), unsafe_allow_html=True)
